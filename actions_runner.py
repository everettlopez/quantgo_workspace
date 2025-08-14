#!/usr/bin/env python3
# QSW Actions Runner — single pass + R2 upload + live heartbeat
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# ---------- Project module path ----------
ROOT = Path(__file__).resolve().parent
PKG  = ROOT / "qsw_go_nogo_v2"
if str(PKG) not in sys.path: sys.path.insert(0, str(PKG))

# Your project imports
from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity

# ---------- Env config ----------
RUNS_ROOT   = Path(os.getenv("QSW_RUNS_DIR", ROOT / "QSW_runs")); RUNS_ROOT.mkdir(parents=True, exist_ok=True)
TRIALS      = int(os.getenv("QSW_TRIALS", "600"))
PHASE_LEN   = int(os.getenv("QSW_PHASE_LEN", "8"))
MAX_STEPS   = int(os.getenv("QSW_MAX_STEPS", "30"))
BATCH_RUN   = int(os.getenv("QSW_BATCH", "256"))
MAX_RETRIES = int(os.getenv("QSW_MAX_RETRIES", "12"))
BACKOFF     = float(os.getenv("QSW_BACKOFF", "2.0"))
PURE_QRNG   = os.getenv("QSW_PURE", "1") == "1"
PROVIDERS   = [p.strip() for p in os.getenv("QSW_QRNG_PROVIDERS", "anu").split(",") if p.strip()]
CONDITIONS  = [c.strip() for c in os.getenv("QSW_CONDITIONS", "quantum").split(",") if c.strip()]

# Cap patience on CI (0=infinite; 1500≈25min is safer)
os.environ["QSW_PURE_MAX_WAIT"] = os.getenv("QSW_PURE_MAX_WAIT", "1500")

# ---------- R2 (S3-compatible) ----------
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION       = os.getenv("S3_REGION", "auto")
S3_BUCKET       = os.getenv("S3_BUCKET")
S3_PREFIX       = os.getenv("S3_PREFIX", "autocollect")
AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")

def _r2():
    import boto3
    from botocore.client import Config
    endpoint = (S3_ENDPOINT_URL or "").strip().rstrip("/")  # trim spaces + no trailing slash
    if endpoint and not endpoint.startswith("http"):
        endpoint = "https://" + endpoint
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION or "auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},   # works best with R2
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )

# ---------- Heartbeat helpers ----------
HEARTBEAT_STABLE = f"{S3_PREFIX.rstrip('/')}/_status/latest.json" if S3_PREFIX else "_status/latest.json"
HEARTBEAT_INPROG = f"{S3_PREFIX.rstrip('/')}/_status/in_progress.json" if S3_PREFIX else "_status/in_progress.json"

def r2_put_json(obj: dict, key: str):
    if not (S3_ENDPOINT_URL and S3_BUCKET and AWS_ACCESS_KEY and AWS_SECRET_KEY):
        return
    s3 = _r2()
    import time as _t
    payload = dict(obj)
    payload["ts_utc"] = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(payload, indent=2).encode(), ContentType="application/json")

def heartbeat(payload: dict, in_progress=True):
    r2_put_json(payload, HEARTBEAT_INPROG if in_progress else HEARTBEAT_STABLE)

# ---------- utils ----------
def _safe(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in "-_,")

def _new_out_dir(label="actions"):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d  = RUNS_ROOT / f"{ts}_v2.7_actions_{_safe(label)}"
    (d / "raw").mkdir(parents=True, exist_ok=True)
    return d

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_bar(df, col, title, ylabel, path):
    fig, ax = plt.subplots()
    ax.bar(df["condition"], df[col])
    ax.set_title(title); ax.set_xlabel("Condition"); ax.set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def r2_upload_dir(local_dir: Path):
    if not (S3_ENDPOINT_URL and S3_BUCKET and AWS_ACCESS_KEY and AWS_SECRET_KEY):
        print("[R2] Skipping upload (missing S3_* env).")
        return 0, [("ALL","Missing S3_* env")]
    s3 = _r2()
    uploaded, skipped = 0, []
    for p in local_dir.rglob("*"):
        if p.is_dir(): continue
        key = f"{S3_PREFIX.rstrip('/')}/{local_dir.name}/{p.relative_to(local_dir).as_posix()}"
        try:
            s3.upload_file(str(p), S3_BUCKET, key)
            uploaded += 1
        except Exception as e:
            skipped.append((key, str(e)))
    print(f"[R2] Uploaded {uploaded} files → r2://{S3_BUCKET}/{S3_PREFIX}/{local_dir.name}/")
    if skipped: print("[R2] Skipped:", skipped[:10])
    return uploaded, skipped

# ---------- core ----------
def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  quantum_only=True, warmup=True,
                  save_trials=True, save_gates=False):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []
    trial_rows = []

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
        import time as _t; _t.sleep(5)

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
                    time_to_escape = t+1
                if w == task.goal:
                    success = True; break

        if success: successes += 1
        tt = time_to_escape if time_to_escape is not None else max_steps
        times.append(tt)
        sequences.append(tuple(actions))
        late_outcomes.append(1 if success else 0)

        if save_trials:
            trial_rows.append({"condition": condition, "trial_idx": i,
                               "success": int(success), "time_to_escape": int(tt),
                               "seq_len": len(actions)})

        # heartbeat every 50 trials
        if (i % 50) == 0:
            try:
                heartbeat({"phase":"running","condition":condition,"trial_idx":i,"n_trials":n_trials}, in_progress=True)
            except Exception:
                pass

    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates); bins = np.linspace(0,1,6)
    dig = np.digitize(eg, bins)-1 if len(eg) else np.array([])
    lo  = np.repeat(late_outcomes, 4)
    L   = min(len(dig), len(lo))
    try: mi = float(mutual_info_score(dig[:L], lo[:L])) if L>1 else 0.0
    except Exception: mi = 0.0

    q_ratio = None
    q_counts = getattr(locals().get('src', object()), "true_quantum_count", None)
    nq_counts= getattr(locals().get('src', object()), "nonquantum_count", None)
    f_counts = getattr(locals().get('src', object()), "fallback_count", None)
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
        "_trial_rows": trial_rows
    }

def main():
    # started heartbeat
    heartbeat({"phase":"started","conditions":CONDITIONS,"pure":PURE_QRNG}, in_progress=True)

    rows = []
    for cond in CONDITIONS:
        rows.append(run_condition(
            n_trials=TRIALS, condition=cond, seed=0,
            phase_len=PHASE_LEN, max_steps=MAX_STEPS,
            batch_run=BATCH_RUN, max_retries=MAX_RETRIES, backoff=BACKOFF,
            providers=PROVIDERS, quantum_only=PURE_QRNG, warmup=True,
            save_trials=True, save_gates=False
        ))

    df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
    out_dir = _new_out_dir(label=("qrng" if PURE_QRNG else "mixed"))
    (out_dir / "results_summary.csv").write_text(df.to_csv(index=False))

    # raw trials
    all_trials = [t for r in rows for t in (r.get("_trial_rows") or [])]
    if all_trials:
        pd.DataFrame(all_trials).to_csv(out_dir / "raw" / "raw_trials.csv.gz",
                                        index=False, compression="gzip")

    # plots
    save_bar(df, "success_rate", "Success Rate — Impasse Escape (v2.7)", "Success Rate", out_dir / "success_rate.png")
    save_bar(df, "avg_time_to_escape", "Avg Time to Escape (lower=better)", "Avg Time to Escape", out_dir / "time_to_escape.png")
    save_bar(df, "diversity_bits", "Diversity (bits)", "Diversity (bits)", out_dir / "diversity_bits.png")
    save_bar(df, "early_gate_success_MI", "Early Gate ↔ Success MI (nats)", "MI (nats)", out_dir / "mi.png")

    # metadata + manifest + status
    meta = {
        "ts_utc": datetime.utcnow().isoformat()+"Z",
        "app_version": "v2.7_actions",
        "pure_qrng": bool(PURE_QRNG),
        "providers": PROVIDERS,
        "params": {"trials": TRIALS, "phase_len": PHASE_LEN, "max_steps": MAX_STEPS,
                   "batch": BATCH_RUN, "retries": MAX_RETRIES, "backoff": BACKOFF,
                   "conditions": CONDITIONS},
        "results": df.to_dict(orient="records")
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

    # final heartbeat (stable)
    final_qr = None
    try:
        if "quantum" in df["condition"].values:
            final_qr = float(df.loc[df["condition"]=="quantum","quantum_ratio"].fillna(0).values[0])
    except Exception:
        pass
    heartbeat({"phase":"finished","run_dir":out_dir.name,
               "files": sum(1 for _ in out_dir.rglob("*") if _.is_file()),
               "quantum_ratio": final_qr}, in_progress=False)

    # upload artifacts
    r2_upload_dir(out_dir)

if __name__ == "__main__":
    main()
