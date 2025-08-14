# streamlit_app.py — QSW v2.7 (saves artifacts + auto-uploads run to Cloudflare R2)
import os, time, json, csv, hashlib, zipfile, shutil
from pathlib import Path
from datetime import datetime
from typing import Dict
from heartbeat_panel import render_heartbeat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="QSW v2.7 — Quantum Go/No-Go", layout="wide")
st.caption(f"UI build: v2.7 • {__file__}")

APP_DIR   = Path(__file__).resolve().parent
TARGET    = APP_DIR / "qsw_go_nogo_v2"
RUNS_ROOT = APP_DIR / "QSW_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------- helpers (saving/artifacts) --------------------
def _safe_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in "-_,")

def _new_run_dir(version="v2.7_streamlit", label=""):
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    lbl = f"_{_safe_name(label)}" if label else ""
    p   = RUNS_ROOT / f"{ts}_{version}{lbl}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _write_json(path: Path, obj: Dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _archive_run(out_dir: Path) -> Path:
    z = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zh:
        for p in out_dir.rglob("*"):
            zh.write(p, p.relative_to(out_dir.parent))
    return z

def save_bar(df, col, title, ylabel, path):
    fig, ax = plt.subplots()
    ax.bar(df["condition"], df[col])
    ax.set_title(title); ax.set_xlabel("Condition"); ax.set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# -------------------- Cloudflare R2 (S3-compatible) --------------------
import boto3, mimetypes
from botocore.client import Config

def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=st.secrets.get("S3_ENDPOINT_URL"),
        aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
        region_name=st.secrets.get("S3_REGION", "auto"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

def r2_upload_dir(local_dir: Path, bucket: str, prefix: str):
    s3 = _r2_client()
    uploaded, skipped = 0, []
    for p in local_dir.rglob("*"):
        if p.is_dir(): 
            continue
        rel = p.relative_to(local_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{local_dir.name}/{rel}"
        ctype, _ = mimetypes.guess_type(p.name)
        extra = {"ContentType": ctype} if ctype else {}
        try:
            s3.upload_file(str(p), bucket, key, ExtraArgs=extra)
            uploaded += 1
        except Exception as e:
            skipped.append((rel, str(e)))
    return uploaded, skipped

# -------------------- providers from Secrets (optional) --------------------
def load_qrng_from_secrets():
    cfg   = st.secrets.get("qrng", {})
    provs = cfg.get("providers", st.secrets.get("QSW_QRNG_PROVIDERS", "anu").split(","))
    os.environ["QSW_QRNG_PROVIDERS"] = ",".join([p.strip() for p in provs if p.strip()])

    if "anu" in cfg:
        os.environ["QSW_ANU_URL"]      = cfg["anu"].get("url", "https://qrng.anu.edu.au/API/jsonI.php")
        os.environ["QSW_ANU_DATA_KEY"] = cfg["anu"].get("data_key", "data")

    for name in ("idq","json"):
        if name in cfg:
            sub = cfg[name]
            os.environ["QSW_JSON_URL"]         = sub.get("url", "")
            os.environ["QSW_JSON_DATA_KEY"]    = sub.get("data_key", "data")
            os.environ["QSW_JSON_IS_QUANTUM"]  = "true" if sub.get("is_quantum", True) else "false"
            if sub.get("auth_env"):
                os.environ["QSW_JSON_AUTH_ENV"] = sub["auth_env"]
                if sub["auth_env"] in st.secrets:
                    os.environ[sub["auth_env"]] = str(st.secrets[sub["auth_env"]])
            os.environ["QSW_JSON_AUTH_HEADER"] = sub.get("auth_header", "Authorization")
            os.environ["QSW_JSON_AUTH_PREFIX"] = sub.get("auth_prefix", "Bearer")
            break
    return [p.strip() for p in provs if p.strip()]

# -------------------- import your project modules (soft) --------------------
has_modules = True
try:
    import sys
    if str(TARGET) not in sys.path: sys.path.insert(0, str(TARGET))
    from importlib import reload
    import qrng_client_v2, agent_workspace_hebbs, tasks_impasse
    reload(qrng_client_v2); reload(agent_workspace_hebbs); reload(tasks_impasse)
    from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
    from agent_workspace_hebbs import WorkspaceAgentHebb
    from tasks_impasse import ImpasseEscapeTask, shannon_diversity
    from sklearn.metrics import mutual_info_score
except Exception as e:
    st.error(f"Import error — add files into qsw_go_nogo_v2/: {e}")
    has_modules = False

# -------------------- UI --------------------
st.title("QSW — Quantum Go/No-Go")
PROVIDER_OPTIONS = load_qrng_from_secrets() if has_modules else ["anu"]

st.sidebar.header("Providers & Purity")
prov_choices = st.sidebar.multiselect("Quantum providers (priority order)",
                                      options=PROVIDER_OPTIONS, default=PROVIDER_OPTIONS)
quantum_only  = st.sidebar.checkbox("Pure QRNG (no PRNG fallback)", True)

st.sidebar.header("Presets")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Short"):   st.session_state.update(dict(trials=250, phase_len=6,  max_steps=30, batch_run=256, max_retries=6,  backoff=1.8))
if c2.button("Standard"):st.session_state.update(dict(trials=600, phase_len=8,  max_steps=30, batch_run=512, max_retries=8,  backoff=1.8))
if c3.button("Deep"):    st.session_state.update(dict(trials=1500,phase_len=12, max_steps=30, batch_run=512, max_retries=10, backoff=2.0))

st.sidebar.header("Parameters")
trials      = st.sidebar.number_input("Trials per condition", min_value=50,  value=st.session_state.get("trials", 600), step=50)
phase_len   = st.sidebar.number_input("Phase length",        min_value=3,   value=st.session_state.get("phase_len", 8))
max_steps   = st.sidebar.number_input("Max steps per trial", min_value=10,  value=st.session_state.get("max_steps", 30))
batch_run   = st.sidebar.number_input("Runtime batch size",  min_value=64,  value=st.session_state.get("batch_run", 512), step=64)
max_retries = st.sidebar.number_input("Max retries/refill",  min_value=3,   value=st.session_state.get("max_retries", 8))
backoff     = st.sidebar.number_input("Backoff factor",      min_value=1.0, value=st.session_state.get("backoff", 1.8), step=0.1, format="%.1f")
min_qratio  = st.sidebar.slider("Min acceptable quantum ratio", 0.0, 1.0, 0.8, 0.05)
warmup_toggle = st.sidebar.checkbox("Warm up quantum source", True)

st.sidebar.header("Per-trial logging")
save_trials = st.sidebar.checkbox("Save per-trial details", True)
save_gates  = st.sidebar.checkbox("Also save full gate sequences", False)

st.write("---")
st.subheader("Run Experiment")
cA, cB, cC = st.columns(3)
use_quantum = cA.checkbox("Run QUANTUM", True)
use_pseudo  = cB.checkbox("Run PSEUDO (control)", False)
use_deter   = cC.checkbox("Run DETERMINISTIC (control)", False)

# -------------------- live ratio panel --------------------
def make_ratio_panel():
    box = st.container(); cols = box.columns(3)
    ratio_metric, q_metric, fb_metric = cols[0].metric, cols[1].metric, cols[2].metric
    chart = box.line_chart({"quantum_ratio": []})
    def update(src):
        q  = getattr(src, "true_quantum_count", 0)
        nq = getattr(src, "nonquantum_count", 0)
        fb = getattr(src, "fallback_count", 0)
        tot = q + nq + fb
        r = (q / tot) if tot else 0.0
        ratio_metric("Quantum Ratio", f"{r:.3f}")
        q_metric("Quantum Draws", f"{int(q):,}")
        fb_metric("Fallback Draws", f"{int(fb):,}")
        chart.add_rows({"quantum_ratio": [r]})
    return update

# -------------------- main run --------------------
def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  ratio_cb=None, quantum_only=True, warmup=True,
                  save_trials=True, save_gates=False):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes  = [], []
    trial_rows, gate_rows       = [], []

    # source
    if condition == "quantum":
        src = QRNGClient('quantum', batch_size=batch_run, max_retries=max_retries,
                         backoff=backoff, providers=providers)
        if quantum_only and hasattr(src, "set_quantum_only"):
            src.set_quantum_only(True)
    elif condition == "pseudo":
        src = QRNGClient('pseudo')
    else:
        src = QRNGClient('deterministic')

    if condition == "quantum" and warmup:
        for _ in range(50): _ = src.next()
        time.sleep(10)

    def next_gate():
        v = float(src.next()) * 1.5
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    update = ratio_cb if callable(ratio_cb) else (lambda *_: None)
    prog = st.progress(0, text=f"{condition}: starting…")

    for i in range(1, n_trials + 1):
        agent = WorkspaceAgentHebb(seed=int(rng.integers(0, 1_000_000_000)))
        task  = ImpasseEscapeTask(seed=int(rng.integers(0, 1_000_000_000)), phase_len=phase_len)

        actions, time_to_escape, success = [], None, False
        gates_this_trial = [] if save_gates else None

        for t in range(max_steps):
            ctx = tuple(actions[-3:]) if actions else None
            g = next_gate()
            if gates_this_trial is not None: gates_this_trial.append(g)
            if t < 4: early_gates.append(g)
            a, probs = agent.step(g, ctx)
            actions.append(a)
            if len(actions) >= task.phase_len:
                w = actions[-task.phase_len:]
                if time_to_escape is None and w[:3]==task.prefix and w!=task.decoy and w==task.goal:
                    time_to_escape = t+1
                if w == task.goal:
                    success = True; break

        if success: successes += 1
        tt = time_to_escape if time_to_escape is not None else max_steps
        times.append(tt)
        seq = tuple(actions); sequences.append(seq)
        late_outcomes.append(1 if success else 0)

        if save_trials:
            trial_rows.append({"condition": condition, "trial_idx": i,
                               "success": int(success), "time_to_escape": int(tt),
                               "seq_len": len(seq)})
        if save_gates and gates_this_trial is not None:
            gate_rows.append({"condition": condition, "trial_idx": i,
                              "gates": ",".join(f"{x:.3f}" for x in gates_this_trial)})

        if i % max(1, n_trials // 20) == 0 or i == n_trials:
            prog.progress(int(100*i/n_trials), text=f"{condition}: {i}/{n_trials}")
        if condition == "quantum":
            step = max(1, n_trials // 10)
            if (i % step == 0) or (i == n_trials): update(src)

    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates); bins = np.linspace(0,1,6)
    dig = np.digitize(eg, bins)-1 if len(eg) else np.array([])
    lo  = np.repeat(late_outcomes, 4)
    L   = min(len(dig), len(lo))
    try: mi = float(mutual_info_score(dig[:L], lo[:L])) if L>1 else 0.0
    except Exception: mi = 0.0

    q_ratio = None
    q_counts = getattr(src, "true_quantum_count", None)
    nq_counts = getattr(src, "nonquantum_count", None)
    f_counts  = getattr(src, "fallback_count", None)
    if condition == "quantum":
        tot = sum(x for x in [q_counts, nq_counts, f_counts] if isinstance(x, int))
        q_ratio = (q_counts / tot) if (isinstance(q_counts, int) and tot>0) else None

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
        "_gate_rows": gate_rows
    }

# -------------------- run button --------------------
start_disabled = not has_modules
if start_disabled:
    st.warning("Missing modules in qsw_go_nogo_v2/. Add files and rerun.")

if st.button("Start Run", disabled=start_disabled):
    conditions = []
    if use_quantum: conditions.append("quantum")
    if use_pseudo:  conditions.append("pseudo")
    if use_deter:   conditions.append("deterministic")
    if not conditions:
        st.error("Select at least one condition.")
    else:
        ratio_cb = make_ratio_panel() if ("quantum" in conditions) else None
        rows = []
        for cond in conditions:
            st.subheader(f"Running: {cond}")
            rows.append(run_condition(
                n_trials=trials, condition=cond, seed=0,
                phase_len=phase_len, max_steps=max_steps,
                batch_run=batch_run, max_retries=max_retries, backoff=backoff,
                providers=prov_choices, ratio_cb=ratio_cb,
                quantum_only=quantum_only, warmup=warmup_toggle,
                save_trials=save_trials, save_gates=save_gates
            ))

        # summary
        df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
        st.write("### Run Summary")
        st.dataframe(df, use_container_width=True)

        # advisory
        if "quantum" in df["condition"].values:
            qr = float(df.loc[df["condition"]=="quantum","quantum_ratio"].fillna(0).values[0])
            if qr >= min_qratio: st.success(f"Quantum source OK (quantum_ratio ≈ {qr:.3f}).")
            elif qr >= 0.5:      st.warning(f"Quantum partially degraded (quantum_ratio = {qr:.3f}).")
            else:                st.error(f"Quantum heavily degraded (quantum_ratio = {qr:.3f}).")

        # save organized artifacts
        label   = "qrng" if use_quantum and not (use_pseudo or use_deter) else "mixed"
        out_dir = _new_run_dir(label=label)

        csv_path = out_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)

        all_trials, all_gates = [], []
        for r in rows:
            if r.get("_trial_rows"): all_trials.extend(r["_trial_rows"])
            if r.get("_gate_rows"):  all_gates.extend(r["_gate_rows"])

        if all_trials:
            (out_dir / "raw").mkdir(exist_ok=True)
            pd.DataFrame(all_trials).to_csv(out_dir / "raw" / "raw_trials.csv.gz",
                                            index=False, compression="gzip")
        if all_gates:
            (out_dir / "raw").mkdir(exist_ok=True)
            pd.DataFrame(all_gates).to_csv(out_dir / "raw" / "raw_gates.csv.gz",
                                           index=False, compression="gzip")

        p1 = out_dir / "success_rate.png"
        p2 = out_dir / "time_to_escape.png"
        p3 = out_dir / "diversity_bits.png"
        p4 = out_dir / "mi.png"
        save_bar(df, "success_rate", "Success Rate — Impasse Escape (v2.7)", "Success Rate", p1)
        save_bar(df, "avg_time_to_escape", "Avg Time to Escape (lower=better)", "Avg Time to Escape", p2)
        save_bar(df, "diversity_bits", "Diversity (bits)", "Diversity (bits)", p3)
        save_bar(df, "early_gate_success_MI", "Early Gate ↔ Success MI (nats)", "MI (nats)", p4)

        meta = {
            "ts_utc": datetime.utcnow().isoformat()+"Z",
            "app_version": "v2.7_streamlit",
            "providers": prov_choices, "quantum_only": bool(quantum_only),
            "logging": {"save_trials": bool(save_trials), "save_gates": bool(save_gates)},
            "params": {"trials": int(trials), "phase_len": int(phase_len), "max_steps": int(max_steps),
                       "batch": int(batch_run), "retries": int(max_retries), "backoff": float(backoff),
                       "warmup": bool(warmup_toggle),
                       "conditions": {"quantum": bool(use_quantum), "pseudo": bool(use_pseudo), "deterministic": bool(use_deter)}},
            "results": df.to_dict(orient="records")
        }
        _write_json(out_dir / "metadata.json", meta)

        manifest = {}
        for p in [csv_path, p1, p2, p3, p4, out_dir / "metadata.json"]:
            manifest[p.name] = {"sha256": _file_sha256(p), "bytes": p.stat().st_size}
        if (out_dir / "raw" / "raw_trials.csv.gz").exists():
            rp = out_dir / "raw" / "raw_trials.csv.gz"
            manifest[rp.name] = {"sha256": _file_sha256(rp), "bytes": rp.stat().st_size}
        if (out_dir / "raw" / "raw_gates.csv.gz").exists():
            gp = out_dir / "raw" / "raw_gates.csv.gz"
            manifest[gp.name] = {"sha256": _file_sha256(gp), "bytes": gp.stat().st_size}
        _write_json(out_dir / "manifest.json", manifest)

        zip_path = _archive_run(out_dir)

        st.success(f"Artifacts saved: {out_dir.name}")
        c1, c2 = st.columns(2)
        with open(csv_path, "rb") as f: c1.download_button("results_summary.csv", f, file_name=csv_path.name, mime="text/csv")
        with open(zip_path, "rb") as f: c2.download_button("full_run.zip", f, file_name=zip_path.name, mime="application/zip")

        # -------------------- AUTO-UPLOAD TO CLOUDFLARE R2 --------------------
        try:
            bucket = st.secrets.get("S3_BUCKET")
            prefix = st.secrets.get("S3_PREFIX", "gsw/runs")
            if bucket and st.secrets.get("S3_ENDPOINT_URL"):
                with st.spinner(f"Uploading {out_dir.name} to R2…"):
                    uploaded, skipped = r2_upload_dir(out_dir, bucket, prefix)
                st.info(f"☁️ R2 upload: {uploaded} files → r2://{bucket}/{prefix}/{out_dir.name}/")
                if skipped:
                    st.warning("Some files skipped:\n" + "\n".join([f"- {n}: {err}" for n, err in skipped[:10]]))
            else:
                st.warning("R2 upload skipped — check S3_* secrets.")
        except Exception as e:
            st.error(f"R2 auto-upload failed: {e}")
        # ---------------------------------------------------------------------
