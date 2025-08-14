# streamlit_app.py — QSW v2.8 (single-folder runs + ordered charts + optional R2 upload)

import os, io, json, time, hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If you used the separate heartbeat panel earlier, you can keep showing it.
# (Safe if the file isn't present — just set SHOW_HEARTBEAT=False below.)
try:
    from heartbeat_panel import render_heartbeat
    HAS_HEARTBEAT = True
except Exception:
    HAS_HEARTBEAT = False

# ---- Optional R2 support for uploads from the app (not required for local only) ----
try:
    import boto3
    from botocore.client import Config
    HAS_BOTO = True
except Exception:
    HAS_BOTO = False

# ============= Display/plot ordering (stable across runs) =============
CONDITION_ORDER = ["quantum", "pseudo", "deterministic"]

def sort_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
        df = df.sort_values("condition")
    return df

# ============= Project imports =============
ROOT = Path(__file__).resolve().parent
PKG  = ROOT / "qsw_go_nogo_v2"
if str(PKG) not in os.sys.path:
    os.sys.path.insert(0, str(PKG))

from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity
from sklearn.metrics import mutual_info_score

# ============= Single-folder naming (matches actions_runner.py) =============
RUNS_ROOT = ROOT / "QSW_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

def _safe(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in "-_,")

def _new_out_dir(conditions, label=None):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = os.getenv("QSW_RUN_NAME") or (label or "-".join(conditions))
    d = RUNS_ROOT / f"{ts}_v2.8_streamlit_{_safe(run_name)}"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ============= Optional R2 helpers =============
def _norm_ep(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    return raw if raw.startswith("http") else ("https://" + raw) if raw else ""

def _r2_client_from_secrets():
    if not (HAS_BOTO and "S3_ENDPOINT_URL" in st.secrets):
        return None
    try:
        return boto3.client(
            "s3",
            endpoint_url=_norm_ep(st.secrets["S3_ENDPOINT_URL"]),
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            region_name=st.secrets.get("S3_REGION", "auto"),
            config=Config(signature_version="s3v4",
                          s3={"addressing_style": "virtual"},
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
    except Exception as e:
        st.warning(f"R2 client init failed (non-fatal): {e}")
        return None

def r2_upload_dir(local_dir: Path):
    s3 = _r2_client_from_secrets()
    if not s3:
        return 0, [("ALL", "No R2 credentials in secrets.toml or boto3 missing")]
    bucket = st.secrets.get("S3_BUCKET", "").strip()
    prefix = st.secrets.get("S3_PREFIX", "autocollect").strip().rstrip("/")
    uploaded, skipped = 0, []
    for p in local_dir.rglob("*"):
        if p.is_dir(): continue
        key = f"{prefix}/{local_dir.name}/{p.relative_to(local_dir).as_posix()}" if prefix else f"{local_dir.name}/{p.relative_to(local_dir).as_posix()}"
        try:
            s3.upload_file(str(p), bucket, key)
            uploaded += 1
        except Exception as e:
            skipped.append((key, str(e)))
    return uploaded, skipped

# ============= Core experiment (runs all conditions into ONE folder) =============
def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  quantum_only=True, warmup=True, save_trials=True):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []
    trial_rows = []

    # RNG source per condition
    if condition == "quantum":
        src = QRNGClient('quantum', batch_size=batch_run, max_retries=max_retries,
                         backoff=backoff, providers=providers)
        if quantum_only and hasattr(src, "set_quantum_only"):
            src.set_quantum_only(True)
    elif condition == "pseudo":
        src = QRNGClient('pseudo')
    else:
        src = QRNGClient('deterministic')

    # QRNG warmup
    if condition == "quantum" and warmup:
        for _ in range(50): _ = src.next()
        time.sleep(5)

    def next_gate():
        v = float(src.next()) * 1.5
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    for i in range(1, n_trials + 1):
        agent = WorkspaceAgentHebb(seed=int(rng.integers(0, 1_000_000_000)))
        task  = ImpasseEscapeTask(seed=int(rng.integers(0, 1_000_000_000)), phase_len=phase_len)

        actions, time_to_escape, success = [], None, False

        for t in range(max_steps):
            ctx = tuple(actions[-3:]) if actions else None
            g = next_gate()
            if t < 4: early_gates.append(g)

            a, probs = agent.step(g, ctx)
            actions.append(a)

            if len(actions) >= task.phase_len:
                w = actions[-task.phase_len:]
                if time_to_escape is None and w[:3]==task.prefix and w!=task.decoy and w==task.goal:
                    time_to_escape = t + 1
                if w == task.goal:
                    success = True
                    break

        if success: successes += 1
        tt = time_to_escape if time_to_escape is not None else max_steps
        times.append(tt)
        sequences.append(tuple(actions))
        late_outcomes.append(1 if success else 0)

        if save_trials:
            trial_rows.append({
                "condition": condition,
                "trial_idx": i,
                "success": int(success),
                "time_to_escape": int(tt),
                "seq_len": len(actions)
            })

    # summary metrics
    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates)
    bins = np.linspace(0, 1, 6)
    dig = np.digitize(eg, bins) - 1 if len(eg) else np.array([])
    lo  = np.repeat(late_outcomes, 4)
    L   = min(len(dig), len(lo))
    try:
        mi = float(mututal_info_score(dig[:L], lo[:L])) if L > 1 else 0.0  # typo? fixed below
    except Exception:
        mi = 0.0

    # (Fix a possible typo in previous versions)
    try:
        mi = float(mutual_info_score(dig[:L], lo[:L])) if L > 1 else 0.0
    except Exception:
        pass

    q_ratio = None
    q_counts  = getattr(locals().get('src', object()), "true_quantum_count", None)
    nq_counts = getattr(locals().get('src', object()), "nonquantum_count", None)
    f_counts  = getattr(locals().get('src', object()), "fallback_count", None)
    if condition == "quantum":
        tot = sum(x for x in [q_counts, nq_counts, f_counts] if isinstance(x, int))
        q_ratio = (q_counts / tot) if (isinstance(q_counts, int) and tot > 0) else None

    return {
        "condition": condition,
        "n_trials": n_trials,
        "success_rate": successes / max(1, n_trials),
        "avg_time_to_escape": float(np.mean(times)),
        "diversity_bits": H,
        "unique_seq": int(unique),
        "early_gate_success_MI": mi,
        "quantum_ratio": q_ratio,
        "q_counts": int(q_counts or 0),
        "nq_counts": int(nq_counts or 0),
        "fallback_counts": int(f_counts or 0),
        "_trial_rows": trial_rows,
    }

# ============================== UI ==============================
st.set_page_config(page_title="Quantum Signal Walker — v2.8", layout="wide")
st.title("Quantum Signal Walker — v2.8")
st.caption("Single-folder runs + deterministic bar order + optional R2 upload")

# (Optional) Heartbeat pane
SHOW_HEARTBEAT = True
if SHOW_HEARTBEAT and HAS_HEARTBEAT:
    render_heartbeat("Collector status")

# Sidebar params
st.sidebar.header("Parameters")
trials    = st.sidebar.number_input("Trials per condition", 100, 5000, 600, step=100)
phase_len = st.sidebar.number_input("Phase length", 4, 20, 8, step=1)
max_steps = st.sidebar.number_input("Max steps", 10, 60, 30, step=1)
batch_run = st.sidebar.number_input("QRNG batch size", 32, 4096, 256, step=32)
max_ret   = st.sidebar.number_input("Max QRNG retries", 0, 50, 12, step=1)
backoff   = st.sidebar.number_input("Retry backoff (sec)", 0.5, 10.0, 2.0, step=0.5)

pure_qrng = st.sidebar.checkbox("Pure QRNG (no fallback) for quantum condition", True)
providers = st.sidebar.text_input("Providers (comma separated)", "anu,json")

st.sidebar.markdown("---")
st.sidebar.subheader("Conditions")
c_quantum = st.sidebar.checkbox("quantum", True)
c_pseudo  = st.sidebar.checkbox("pseudo", True)
c_determ  = st.sidebar.checkbox("deterministic", True)

st.sidebar.markdown("---")
run_name = st.sidebar.text_input("Run name tag (optional)", "all")

if st.button("Run experiment"):
    conditions = [c for c, on in [("quantum", c_quantum), ("pseudo", c_pseudo), ("deterministic", c_determ)] if on]
    if not conditions:
        st.error("Select at least one condition.")
        st.stop()

    # run all selected conditions → ONE output folder
    out_dir = _new_out_dir(conditions, label=run_name)
    st.info(f"Output folder: `{out_dir}`")

    rows = []
    t0 = time.time()
    for cond in conditions:
        st.write(f"### Running: `{cond}`")
        rows.append(run_condition(
            n_trials=trials, condition=cond, seed=0,
            phase_len=phase_len, max_steps=max_steps,
            batch_run=batch_run, max_retries=max_ret, backoff=backoff,
            providers=[p.strip() for p in providers.split(",") if p.strip()],
            quantum_only=bool(pure_qrng), warmup=True, save_trials=True
        ))

    # summarize
    df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
    df = sort_results_df(df)

    # save CSV
    (out_dir / "results_summary.csv").write_text(df.to_csv(index=False))

    # raw trials
    all_trials = [t for r in rows for t in (r.get("_trial_rows") or [])]
    if all_trials:
        pd.DataFrame(all_trials).to_csv(out_dir / "raw" / "raw_trials.csv.gz",
                                        index=False, compression="gzip")

    # plots (ordered by condition)
    def _save_bar(df, col, title, ylabel, path):
        fig, ax = plt.subplots()
        ax.bar(df["condition"], df[col])
        ax.set_title(title); ax.set_xlabel("Condition"); ax.set_ylabel(ylabel)
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    _save_bar(df, "success_rate", "Success Rate — Impasse Escape (v2.8)", "Success Rate", out_dir / "success_rate.png")
    _save_bar(df, "avg_time_to_escape", "Avg Time to Escape (lower=better)", "Avg Time to Escape", out_dir / "time_to_escape.png")
    _save_bar(df, "diversity_bits", "Diversity (bits)", "Diversity (bits)", out_dir / "diversity_bits.png")
    _save_bar(df, "early_gate_success_MI", "Early Gate ↔ Success MI (nats)", "MI (nats)", out_dir / "mi.png")

    # metadata + manifest
    meta = {
        "ts_utc": datetime.utcnow().isoformat() + "Z",
        "app_version": "v2.8_streamlit",
        "pure_qrng": bool(pure_qrng),
        "providers": [p.strip() for p in providers.split(",") if p.strip()],
        "params": {"trials": trials, "phase_len": phase_len, "max_steps": max_steps,
                   "batch": batch_run, "retries": max_ret, "backoff": backoff,
                   "conditions": conditions},
        "results": df.to_dict(orient="records"),
        "elapsed_total_sec": int(time.time() - t0)
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    manifest = {}
    for p in [out_dir / "results_summary.csv",
              out_dir / "success_rate.png", out_dir / "time_to_escape.png",
              out_dir / "diversity_bits.png", out_dir / "mi.png",
              out_dir / "metadata.json"]:
        manifest[p.name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    rp = out_dir / "raw" / "raw_trials.csv.gz"
    if rp.exists():
        manifest[rp.name] = {"sha256": _sha256(rp), "bytes": rp.stat().st_size}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # R2 upload (non-fatal)
    with st.spinner("Uploading artifacts to Cloudflare R2 (if configured)…"):
        up, skipped = r2_upload_dir(out_dir)
        if up:
            st.success(f"Uploaded {up} files to R2 under `{st.secrets.get('S3_PREFIX','autocollect')}/{out_dir.name}`")
        if skipped and not up:
            st.info(f"R2 upload skipped: {skipped[0][1]}")

    # Show results
    st.success("Run complete!")
    st.write("### Summary")
    st.dataframe(df)

    st.write("### Plots")
    c1, c2 = st.columns(2)
    c1.image(str(out_dir / "success_rate.png"), caption="Success Rate")
    c2.image(str(out_dir / "time_to_escape.png"), caption="Avg Time to Escape")
    c3, c4 = st.columns(2)
    c3.image(str(out_dir / "diversity_bits.png"), caption="Diversity (bits)")
    c4.image(str(out_dir / "mi.png"), caption="Early Gate ↔ Success MI")

else:
    st.info("Configure parameters on the left, then click **Run experiment**. (All selected conditions will be saved into **one** output folder.)")
