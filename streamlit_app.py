# streamlit_app.py — QSW v2.7 (keeps v2.5 layout, adds Pure QRNG + autosave + per-trial logging)
import os, time, zipfile, shutil, json, csv, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="QSW v2.7 — Quantum Go/No-Go", layout="wide")
st.caption(f"UI build: v2.7 • main={__file__}")

APP_DIR = Path(__file__).resolve().parent
TARGET_DIR = APP_DIR / "qsw_go_nogo_v2"
RUNS_ROOT = APP_DIR / "QSW_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- helpers (saving/artifacts) ----------
# === Cloudflare R2 (S3) helpers ===
import boto3, mimetypes
from botocore.client import Config
from pathlib import Path
import streamlit as st

def _r2_client():
    kwargs = {
        "endpoint_url": st.secrets.get("S3_ENDPOINT_URL"),
        "aws_access_key_id": st.secrets.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": st.secrets.get("AWS_SECRET_ACCESS_KEY"),
        "region_name": st.secrets.get("S3_REGION", "auto"),
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    }
    return boto3.client("s3", **{k: v for k, v in kwargs.items() if v})

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
            # private by default; share with presigned URLs if needed
            s3.upload_file(str(p), bucket, key, ExtraArgs=extra)
            uploaded += 1
        except Exception as e:
            skipped.append((rel, str(e)))
    return uploaded, skipped

def r2_presign(bucket: str, key: str, expires_seconds: int = 3600):
    s3 = _r2_client()
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds
    )
# === end R2 helpers ===
def _safe_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in "-_,")

def _new_run_dir(version_tag="v2.7_streamlit", label=""):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    lbl = f"_{_safe_name(label)}" if label else ""
    p = RUNS_ROOT / f"{ts}_{version_tag}{lbl}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _write_json(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _archive_run(out_dir: Path) -> Path:
    zpath = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            z.write(p, p.relative_to(out_dir.parent))
    return zpath

# ---------- providers from Secrets (non-fatal) ----------
def load_qrng_from_secrets():
    cfg = st.secrets.get("qrng", {})
    provs = cfg.get("providers", ["anu"])
    os.environ["QSW_QRNG_PROVIDERS"] = ",".join(provs)

    if "anu" in cfg:
        os.environ["QSW_ANU_URL"] = cfg["anu"].get("url", "https://qrng.anu.edu.au/API/jsonI.php")
        os.environ["QSW_ANU_DATA_KEY"] = cfg["anu"].get("data_key", "data")

    for name in ("idq", "json"):
        if name in cfg:
            sub = cfg[name]
            os.environ["QSW_JSON_URL"] = sub.get("url", "")
            os.environ["QSW_JSON_DATA_KEY"] = sub.get("data_key", "data")
            os.environ["QSW_JSON_IS_QUANTUM"] = "true" if sub.get("is_quantum", True) else "false"
            if sub.get("auth_env"):
                os.environ["QSW_JSON_AUTH_ENV"] = sub["auth_env"]
                if sub["auth_env"] in st.secrets:
                    os.environ[sub["auth_env"]] = str(st.secrets[sub["auth_env"]])
            os.environ["QSW_JSON_AUTH_HEADER"] = sub.get("auth_header", "Authorization")
            os.environ["QSW_JSON_AUTH_PREFIX"] = sub.get("auth_prefix", "Bearer")
            break
    return provs

# ---------- require project folder but don’t hard-stop the whole UI ----------
def ensure_project_folder():
    if TARGET_DIR.is_dir():
        return True
    st.warning("Project folder 'qsw_go_nogo_v2' not found. Upload a ZIP containing that folder.")
    up = st.file_uploader("Upload project ZIP", type=["zip"])
    if up is not None:
        tmp = APP_DIR / "uploaded_qsw_bundle.zip"
        with open(tmp, "wb") as f: f.write(up.read())
        try:
            with zipfile.ZipFile(tmp, "r") as z: z.extractall(APP_DIR)
            found = None
            for root, _, _ in os.walk(APP_DIR):
                if os.path.basename(root) == "qsw_go_nogo_v2":
                    found = Path(root); break
            if found and found != TARGET_DIR:
                if TARGET_DIR.exists(): shutil.rmtree(TARGET_DIR)
                shutil.move(str(found), str(TARGET_DIR))
            if TARGET_DIR.is_dir():
                st.success("Bundle unpacked. Use the top-right menu → **Rerun**.")
                return True
        except Exception as e:
            st.error(f"Unpack failed: {e}")
    return TARGET_DIR.is_dir()

# ---------- Imports (soft) ----------
ok_proj = ensure_project_folder()
if ok_proj:
    import sys
    if str(TARGET_DIR) not in sys.path: sys.path.insert(0, str(TARGET_DIR))
    from importlib import reload
    try:
        import qrng_client_v2, agent_workspace_hebbs, tasks_impasse
        reload(qrng_client_v2); reload(agent_workspace_hebbs); reload(tasks_impasse)
        from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
        from agent_workspace_hebbs import WorkspaceAgentHebb
        from tasks_impasse import ImpasseEscapeTask, shannon_diversity
        from sklearn.metrics import mutual_info_score
        st.caption("Modules: OK")
        has_modules = True
    except Exception as e:
        st.error(f"Import error: {type(e).__name__}: {e}")
        has_modules = False
else:
    has_modules = False

# ---------- Title ----------
st.title("QSW — Quantum Go/No-Go")

# ---------- Sidebar (providers + purity + params) ----------
PROVIDER_OPTIONS = load_qrng_from_secrets() if has_modules else ["anu"]

st.sidebar.header("Providers & Purity")
prov_choices = st.sidebar.multiselect(
    "Quantum providers (priority order)",
    options=PROVIDER_OPTIONS,
    default=PROVIDER_OPTIONS
)
quantum_only = st.sidebar.checkbox("Pure QRNG (no PRNG fallback)", True)

st.sidebar.header("Presets")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("Short Test"):     st.session_state.update(dict(trials=250, phase_len=6,  max_steps=30, batch_run=256, max_retries=6, backoff=1.8))
if c2.button("Standard Run"):   st.session_state.update(dict(trials=600, phase_len=8,  max_steps=30, batch_run=512, max_retries=8, backoff=1.8))
if c3.button("Deep Scan"):      st.session_state.update(dict(trials=1500,phase_len=12, max_steps=30, batch_run=512, max_retries=10, backoff=2.0))

st.sidebar.header("Parameters")
trials      = st.sidebar.number_input("Trials per condition", min_value=50, value=st.session_state.get("trials", 600), step=50)
phase_len   = st.sidebar.number_input("Phase length", min_value=3,  value=st.session_state.get("phase_len", 8))
max_steps   = st.sidebar.number_input("Max steps per trial", min_value=10, value=st.session_state.get("max_steps", 30))
batch_run   = st.sidebar.number_input("Runtime batch size", min_value=64, value=st.session_state.get("batch_run", 512), step=64)
max_retries = st.sidebar.number_input("Max retries/refill", min_value=3,  value=st.session_state.get("max_retries", 8))
backoff     = st.sidebar.number_input("Backoff factor", min_value=1.0, value=st.session_state.get("backoff", 1.8), step=0.1, format="%.1f")
min_qratio  = st.sidebar.slider("Min acceptable quantum ratio (advisory)", 0.0, 1.0, 0.8, 0.05)
warmup_toggle = st.sidebar.checkbox("Warm up quantum source before run", True)

st.sidebar.header("Per-trial logging")
save_trials = st.sidebar.checkbox("Save per-trial details", True)
save_gates  = st.sidebar.checkbox("Also save full gate sequences (large)", False)

# ---------- Mini ratio panel maker ----------
def make_ratio_panel():
    box = st.container()
    cols = box.columns(3)
    ratio_metric = cols[0].metric; q_metric = cols[1].metric; fb_metric = cols[2].metric
    chart = box.line_chart({"quantum_ratio": []})
    def update_from_client(src):
        q  = getattr(src, "true_quantum_count", 0)
        nq = getattr(src, "nonquantum_count", 0)
        fb = getattr(src, "fallback_count", 0)
        tot = q + nq + fb
        ratio = (q / tot) if tot else 0.0
        ratio_metric("Quantum Ratio", f"{ratio:.3f}")
        q_metric("Quantum Draws", f"{int(q):,}")
        fb_metric("Fallback Draws", f"{int(fb):,}")
        chart.add_rows({"quantum_ratio": [ratio]})
    return update_from_client

# ---------- Core run function (aligned with your v2.5 logic) ----------
def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  ratio_cb=None, quantum_only=True, warmup=True,
                  save_trials=True, save_gates=False):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []

    trial_rows, gate_rows = [], []

    # Build source
    if condition == "quantum":
        src = QRNGClient('quantum', batch_size=batch_run, max_retries=max_retries,
                         backoff=backoff, providers=providers)
        try:
            if quantum_only and hasattr(src, "set_quantum_only"):
                src.set_quantum_only(True)
        except Exception:
            pass
    elif condition == "pseudo":
        src = QRNGClient('pseudo')
    else:
        src = QRNGClient('deterministic')

    # Optional warmup
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
        seq = tuple(actions)
        sequences.append(seq)
        late_outcomes.append(1 if success else 0)

        if save_trials:
            trial_rows.append({
                "condition": condition,
                "trial_idx": i,
                "success": int(success),
                "time_to_escape": int(tt),
                "seq_len": len(seq)
            })
        if save_gates and gates_this_trial is not None:
            gate_rows.append({
                "condition": condition,
                "trial_idx": i,
                "gates": ",".join(f"{x:.3f}" for x in gates_this_trial)
            })

        # UI & live ratio
        if i % max(1, n_trials // 20) == 0 or i == n_trials:
            prog.progress(int(100 * i / n_trials), text=f"{condition}: {i}/{n_trials}")
        if condition == "quantum":
            step = max(1, n_trials // 10)
            if (i % step == 0) or (i == n_trials): update(src)

    # Aggregate metrics
    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates); bins = np.linspace(0,1,6)
    dig = np.digitize(eg, bins)-1 if len(eg) else np.array([])
    lo = np.repeat(late_outcomes, 4)
    L = min(len(dig), len(lo))
    try: mi = float(mutual_info_score(dig[:L], lo[:L])) if L>1 else 0.0
    except Exception: mi = 0.0

    # Telemetry
    q_ratio = None
    q_counts = getattr(src, "true_quantum_count", None)
    nq_counts = getattr(src, "nonquantum_count", None)
    f_counts = getattr(src, "fallback_count", None)
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
        "_gate_rows": gate_rows
    }

# ---------- Run Experiment (keeps your v2.5 flow) ----------
st.write("---")
st.subheader("Run Experiment")
c1, c2, c3 = st.columns(3)
use_quantum = c1.checkbox("Run QUANTUM", True)
use_pseudo  = c2.checkbox("Run PSEUDO (control)", False)
use_deter   = c3.checkbox("Run DETERMINISTIC (control)", False)

start_disabled = not has_modules
if start_disabled:
    st.warning("Missing modules in qsw_go_nogo_v2/. Upload your project ZIP, then Rerun.")

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

        # Summary table (same as last night’s flow)
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        st.write("### Run Summary")
        st.dataframe(df, use_container_width=True)

        # Advisory banner for quantum
        if "quantum" in df["condition"].values:
            qr = float(df.loc[df["condition"]=="quantum","quantum_ratio"].fillna(0).values[0])
            if qr >= min_qratio:
                st.success(f"Quantum source OK (quantum_ratio ≈ {qr:.3f}).")
            elif qr >= 0.5:
                st.warning(f"Quantum source partially degraded (quantum_ratio = {qr:.3f}).")
            else:
                st.error(f"Quantum source heavily degraded (quantum_ratio = {qr:.3f}).")

        # ---- Organized save (keeps your v2.5 outputs & adds structure) ----
        run_label = "qrng" if use_quantum and (not use_pseudo and not use_deter) else "mixed"
        out_dir = _new_run_dir(version_tag="v2.7_streamlit", label=run_label)

        # 1) Save summary CSV
        csv_path = out_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)

        # 2) Per-trial & gates (optional) — compressed
        all_trials, all_gates = [], []
        for r in rows:
            trs = r.get("_trial_rows") or []
            gts = r.get("_gate_rows") or []
            if trs: all_trials.extend(trs)
            if gts: all_gates.extend(gts)
        if all_trials:
            (out_dir / "raw").mkdir(exist_ok=True)
            pd.DataFrame(all_trials).to_csv(out_dir / "raw" / "raw_trials.csv.gz",
                                            index=False, compression="gzip")
        if all_gates:
            (out_dir / "raw").mkdir(exist_ok=True)
            pd.DataFrame(all_gates).to_csv(out_dir / "raw" / "raw_gates.csv.gz",
                                           index=False, compression="gzip")

        # 3) Plots (same placement and names as your v2.5)
        def save_bar(df, col, title, ylabel, path):
            fig, ax = plt.subplots()
            ax.bar(df["condition"], df[col])
            ax.set_title(title); ax.set_xlabel("Condition"); ax.set_ylabel(ylabel)
            fig.savefig(path, bbox_inches="tight"); plt.close(fig)

        p1 = out_dir / "success_rate.png"
        p2 = out_dir / "time_to_escape.png"
        p3 = out_dir / "diversity_bits.png"
        p4 = out_dir / "mi.png"
        def _save(col, title, ylabel, path): save_bar(df, col, title, ylabel, str(path))
        _save("success_rate", "Success Rate — Impasse Escape (v2.7)", "Success Rate", p1)
        _save("avg_time_to_escape", "Avg Time to Escape (lower=better)", "Avg Time to Escape", p2)
        _save("diversity_bits", "Diversity (bits)", p3.name.replace(".png",""), p3)  # ylabel shown as title for compactness
        _save("early_gate_success_MI", "Early Gate ↔ Success MI (nats)", "MI (nats)", p4)

        # Plots shown in the same way as last night
        st.write("### Plots")
        c1, c2 = st.columns(2); c1.image(str(p1)); c2.image(str(p2))
        c3, c4 = st.columns(2); c3.image(str(p3)); c4.image(str(p4))

        # 4) Metadata & manifest
        meta = {
            "ts_utc": datetime.utcnow().isoformat()+"Z",
            "app_version": "v2.7_streamlit",
            "providers": prov_choices,
            "quantum_only": bool(quantum_only),
            "logging": {"save_trials": bool(save_trials), "save_gates": bool(save_gates)},
            "params": {
                "trials": int(trials), "phase_len": int(phase_len),
                "max_steps": int(max_steps), "batch": int(batch_run),
                "retries": int(max_retries), "backoff": float(backoff),
                "warmup": bool(warmup_toggle),
                "conditions": {"quantum": bool(use_quantum), "pseudo": bool(use_pseudo), "deterministic": bool(use_deter)}
            },
            "results": df.to_dict(orient="records")
        }
        _write_json(out_dir / "metadata.json", meta)

        manifest = {}
        def add_manifest(p: Path):
            if p.exists(): manifest[p.name] = {"sha256": _file_sha256(p), "bytes": p.stat().st_size}
        for p in [csv_path, p1, p2, p3, p4, out_dir / "metadata.json"]:
            add_manifest(p)
        if (out_dir / "raw" / "raw_trials.csv.gz").exists(): add_manifest(out_dir / "raw" / "raw_trials.csv.gz")
        if (out_dir / "raw" / "raw_gates.csv.gz").exists():  add_manifest(out_dir / "raw" / "raw_gates.csv.gz")
        _write_json(out_dir / "manifest.json", manifest)

        # 5) Archive + Downloads (keeps your v2.5 “downloads” section feel)
        zip_path = _archive_run(out_dir)

        st.write("### Downloads")
        with open(csv_path, "rb") as f:
            st.download_button("results_summary.csv", f, file_name=csv_path.name, mime="text/csv")
        with open(zip_path, "rb") as f:
            st.download_button("full_run.zip", f, file_name=zip_path.name, mime="application/zip")
            # --- Backup to R2 (Cloudflare) ---
st.write("### Backup")
if st.button("Backup this run to Cloudflare R2"):
    with st.spinner("Uploading artifacts to R2…"):
        bucket = st.secrets.get("S3_BUCKET")
        prefix = st.secrets.get("S3_PREFIX", "gsw/runs")
        if not (bucket and st.secrets.get("S3_ENDPOINT_URL")):
            st.error("Missing R2 secrets (S3_BUCKET / S3_ENDPOINT_URL / AWS keys).")
        else:
            try:
                uploaded, skipped = r2_upload_dir(out_dir, bucket, prefix)
                st.success(f"Uploaded {uploaded} files to r2://{bucket}/{prefix}/{out_dir.name}/")
                if skipped:
                    st.warning("Some files skipped:\n" + "\n".join([f"- {n}: {err}" for n, err in skipped[:10]]))

                # Handy quick links (valid 1 hour)
                keys = [
                    f"{prefix}/{out_dir.name}/results_summary.csv",
                    f"{prefix}/{out_dir.name}/metadata.json",
                    f"{prefix}/{out_dir.name}/manifest.json",
                    f"{prefix}/{out_dir.name}/success_rate.png",
                    f"{prefix}/{out_dir.name}/time_to_escape.png",
                ]
                st.write("**Quick presigned links (1h):**")
                for k in keys:
                    try:
                        url = r2_presign(bucket, k, 3600)
                        st.write(f"- [{k.split('/')[-1]}]({url})")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"R2 backup failed: {e}")
                # after you finish saving artifacts to `out_dir`:
st.session_state['last_out_dir'] = str(out_dir)  # <-- remember the folder for backup
st.write("### Cloudflare R2 Backup")
if st.button("Backup last run to R2"):
    p = st.session_state.get('last_out_dir')
    if not p:
        st.warning("No completed run found this session. Run the experiment first.")
    else:
        out_dir = Path(p)
        bucket = st.secrets.get("S3_BUCKET")
        prefix = st.secrets.get("S3_PREFIX", "gsw/runs")
        if not (bucket and st.secrets.get("S3_ENDPOINT_URL") and
                st.secrets.get("AWS_ACCESS_KEY_ID") and st.secrets.get("AWS_SECRET_ACCESS_KEY")):
            st.error("Missing R2 secrets (S3_ENDPOINT_URL, S3_BUCKET, AWS keys).")
        else:
            with st.spinner(f"Uploading {out_dir.name} to R2…"):
                try:
                    uploaded, skipped = r2_upload_dir(out_dir, bucket, prefix)
                    st.success(f"Uploaded {uploaded} files to r2://{bucket}/{prefix}/{out_dir.name}/")
                    if skipped:
                        st.warning("Some files skipped:\n" + "\n".join([f"- {n}: {err}" for n, err in skipped[:10]]))
                except Exception as e:
                    st.error(f"R2 backup failed: {e}")
