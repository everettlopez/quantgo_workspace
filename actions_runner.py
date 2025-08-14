#!/usr/bin/env python3
# QSW Actions Runner: one single run, then exit. Sequential-safe via GitHub Actions concurrency.
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# Project module path
ROOT = Path(__file__).resolve().parent
PKG  = ROOT / "qsw_go_nogo_v2"
if str(PKG) not in sys.path: sys.path.insert(0, str(PKG))

# Your project imports (must exist in repo)
from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity

# Config via env
RUNS_ROOT   = Path(os.getenv("QSW_RUNS_DIR", ROOT / "QSW_runs")); RUNS_ROOT.mkdir(parents=True, exist_ok=True)
TRIALS      = int(os.getenv("QSW_TRIALS", "600"))
PHASE_LEN   = int(os.getenv("QSW_PHASE_LEN", "8"))
MAX_STEPS   = int(os.getenv("QSW_MAX_STEPS", "30"))
BATCH_RUN   = int(os.getenv("QSW_BATCH", "256"))
MAX_RETRIES = int(os.getenv("QSW_MAX_RETRIES", "12"))
BACKOFF     = float(os.getenv("QSW_BACKOFF", "2.0"))
PURE_QRNG   = os.getenv("QSW_PURE", "1") == "1"
PROVIDERS   = [p.strip() for p in os.getenv("QSW_QRNG_PROVIDERS", "anu").split(",") if p.strip()]
CONDITIONS  = [c.strip() for c in os.getenv("QSW_CONDITIONS", "quantum,pseudo,deterministic").split(",") if c.strip()]

# Cap pure-mode patience on CI (0 = infinite; 1500s ≈ 25 min is safer for Actions)
os.environ["QSW_PURE_MAX_WAIT"] = os.getenv("QSW_PURE_MAX_WAIT", "1500")

# R2 (S3-compatible)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION       = os.getenv("S3_REGION", "auto")
S3_BUCKET       = os.getenv("S3_BUCKET")
S3_PREFIX       = os.getenv("S3_PREFIX", "gsw/runs")
AWS_ACCESS_KEY  = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY  = os.getenv("AWS_SECRET_ACCESS_KEY")

def _r2():
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

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
        return 0, [("ALL", "Missing S3_* env")]
    s3 = _r2()
    uploaded, skipped = 0, []
    for p in local_dir.rglob("*"):
        if p.is_dir(): continue
        rel = p.relative_to(local_dir).as_posix()
        key = f"{S3_PREFIX.rstrip('/')}/{local_dir.name}/{rel}"
        try:
            s3.upload_file(str(p), S3_BUCKET, key)
            uploaded += 1
        except Exception as e:
            skipped.append((rel, str(e)))
    print(f"[R2] Uploaded {uploaded} files → r2://{S3_BUCKET}/{S3_PREFIX}/{local_dir.name}/")
    if skipped: print("[R2] Skipped:", skipped[:10])
    return uploaded, skipped

def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  quantum_only=True, warmup=True,
                  save_trials=True, save_gates=False):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []
    trial_rows, gate_rows = [], []

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
        "_trial_rows": trial_rows,
        "_gate_rows": gate_rows
    }

def main():
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

    # metadata + manifest
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

    # quick status line for monitoring
    status = {
        "run_dir": out_dir.name,
        "files": sum(1 for _ in out_dir.rglob("*") if _.is_file()),
        "quantum_ratio": float(df.loc[df["condition"]=="quantum","quantum_ratio"].fillna(0).values[0]) if "quantum" in df["condition"].values else None
    }
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))

    # upload
    r2_upload_dir(out_dir)

if __name__ == "__main__":
    main()
