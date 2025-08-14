# streamlit_app.py — QSW v2.5 (self-contained)
# - Live Quantum Ratio panel (no extra draws)
# - Optional Warmup mode
# - Multi-QRNG provider picker from Secrets
# - Uses only in-repo modules: qsw_go_nogo_v2/qrng_client_v2.py, agent_workspace_hebbs.py, tasks_impasse.py

import os, io, time, zipfile, shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Basic setup ----------
st.set_page_config(page_title="QSW v2.5 — Quantum Go/No-Go", layout="wide")
APP_DIR = Path(__file__).resolve().parent
TARGET_DIR = APP_DIR / "qsw_go_nogo_v2"

st.title("QSW v2.5 — Quantum Go/No-Go (Streamlit)")

# ---------- Secrets → env (providers) ----------
def load_qrng_from_secrets():
    cfg = st.secrets.get("qrng", {})
    prov_list = cfg.get("providers", ["anu"])
    os.environ["QSW_QRNG_PROVIDERS"] = ",".join(prov_list)

    # ANU
    if "anu" in cfg:
        os.environ["QSW_ANU_URL"] = cfg["anu"].get("url", "https://qrng.anu.edu.au/API/jsonI.php")
        os.environ["QSW_ANU_DATA_KEY"] = cfg["anu"].get("data_key", "data")

    # First JSON-like source (IDQ or generic)
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

    return prov_list

PROVIDER_OPTIONS = load_qrng_from_secrets()

# ---------- Ensure project files are present (optional ZIP upload) ----------
def unpack_bundle(zpath: Path):
    try:
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(APP_DIR)
        # normalize path if extracted elsewhere
        found = None
        for root, dirs, files in os.walk(APP_DIR):
            if os.path.basename(root) == "qsw_go_nogo_v2":
                found = Path(root); break
        if found and found != TARGET_DIR:
            if TARGET_DIR.exists():
                shutil.rmtree(TARGET_DIR)
            shutil.move(str(found), str(TARGET_DIR))
        return TARGET_DIR.is_dir()
    except Exception as e:
        st.error(f"Unpack failed: {e}")
        return False

if not TARGET_DIR.is_dir():
    st.warning("Project folder 'qsw_go_nogo_v2' not found. Upload a ZIP containing that folder.")
    up = st.file_uploader("Upload project ZIP", type=["zip"])
    if up is not None:
        tmp = APP_DIR / "uploaded_qsw_bundle.zip"
        with open(tmp, "wb") as f:
            f.write(up.read())
        if unpack_bundle(tmp):
            st.success("Bundle unpacked. Use the top-right menu → **Rerun**.")
    st.stop()

# ---------- Import actual in-repo modules ----------
import sys
if str(TARGET_DIR) not in sys.path:
    sys.path.insert(0, str(TARGET_DIR))

from importlib import reload
import qrng_client_v2, agent_workspace_hebbs, tasks_impasse
reload(qrng_client_v2); reload(agent_workspace_hebbs); reload(tasks_impasse)
from qsw_go_nogo_v2.qrng_client_v2 import QRNGClient
from agent_workspace_hebbs import WorkspaceAgentHebb
from tasks_impasse import ImpasseEscapeTask, shannon_diversity
from sklearn.metrics import mutual_info_score

# ---------- Helpers defined here ----------
def save_bar(df, col, title, ylabel, path):
    fig, ax = plt.subplots()
    ax.bar(df["condition"], df[col])
    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# Live panel that shows ratio from an existing QRNGClient (no extra draws)
def make_ratio_panel():
    box = st.container()
    cols = box.columns(3)
    ratio_metric = cols[0].metric
    q_metric     = cols[1].metric
    fb_metric    = cols[2].metric
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

def run_condition(n_trials, condition, seed, phase_len, max_steps,
                  batch_run, max_retries, backoff, providers,
                  ratio_cb=None, warmup=False):
    rng = np.random.default_rng(seed)
    successes, times, sequences = 0, [], []
    early_gates, late_outcomes = [], []

    # Build source
    if condition == "quantum":
        src = QRNGClient('quantum', batch_size=batch_run, max_retries=max_retries,
                         backoff=backoff, providers=providers)
    elif condition == "pseudo":
        src = QRNGClient('pseudo')
    else:
        src = QRNGClient('deterministic')

    # Optional warmup (small draw + pause) — uses SAME client to avoid double-dipping
    if condition == "quantum" and warmup:
        for _ in range(50):  # tiny touch
            _ = src.next()
        time.sleep(15)

    # Gate function (works with or without internal buffer)
    def next_gate():
        v = src.next()
        v = float(v) * 1.5
        return 0.0 if v < 0 else 1.0 if v > 1 else v

    # optional live panel updater (uses the same client; no extra draws)
    update = ratio_cb if callable(ratio_cb) else (lambda *_: None)

    prog = st.progress(0, text=f"{condition}: starting…")
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
        times.append(time_to_escape if time_to_escape is not None else max_steps)
        sequences.append(tuple(actions))
        late_outcomes.append(1 if success else 0)

        # UI pacing
        if i % max(1, n_trials // 20) == 0 or i == n_trials:
            prog.progress(int(100 * i / n_trials), text=f"{condition}: {i}/{n_trials}")
        # Update live ratio panel ~25× per run (no extra draws)
        if condition == "quantum" and (i % max(1, n_trials // 25) == 0 or i == n_trials):
            update(src)

    # Metrics
    H, unique = shannon_diversity(sequences)
    eg = np.array(early_gates); bins = np.linspace(0,1,6)
    dig = np.digitize(eg, bins)-1 if len(eg) else np.array([])
    lo = np.repeat(late_outcomes, 4)
    L = min(len(dig), len(lo))
    try:
        mi = float(mutual_info_score(dig[:L], lo[:L])) if L>1 else 0.0
    except Exception:
        mi = 0.0

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
    }

# ---------- Sidebar controls (safer defaults) ----------
st.sidebar.header("Controls")
prov_choices = st.sidebar.multiselect(
    "Quantum providers (priority order)",
    options=PROVIDER_OPTIONS,
    default=PROVIDER_OPTIONS
)
trials      = st.sidebar.slider("Trials per condition", 100, 2000, 600, 100)
phase_len   = st.sidebar.slider("Phase length", 6, 20, 6, 1)           # safer for QRNG load
max_steps   = st.sidebar.slider("Max steps per trial", 20, 60, 30, 1)
batch_run   = st.sidebar.slider("Runtime batch size", 128, 4096, 256, 128)  # smaller batch
max_retries = st.sidebar.slider("Max retries/refill", 3, 12, 6, 1)
backoff     = st.sidebar.slider("Backoff factor", 1.0, 3.0, 1.8, 0.1)
min_qratio  = st.sidebar.slider("Min acceptable quantum ratio (advisory)", 0.0, 1.0, 0.8, 0.05)
show_live_ratio = st.sidebar.checkbox("Show live quantum ratio during run (no extra draws)", True)
warmup_toggle   = st.sidebar.checkbox("Warm up quantum source before run", True)

# ---------- Run ----------
if st.button("Run QSW v2.5"):
    conditions = []
    if st.sidebar.checkbox("Run QUANTUM condition", True): conditions.append("quantum")
    if st.sidebar.checkbox("Run PSEUDO condition", True):  conditions.append("pseudo")
    if st.sidebar.checkbox("Run DETERMINISTIC condition", True): conditions.append("deterministic")
    if not conditions:
        st.error("Select at least one condition."); st.stop()

    ratio_cb = make_ratio_panel() if (show_live_ratio and "quantum" in conditions) else None

    rows = []
    for cond in conditions:
        st.subheader(f"Running: {cond}")
        rows.append(run_condition(trials, cond, seed=0, phase_len=phase_len, max_steps=max_steps,
                                  batch_run=batch_run, max_retries=max_retries, backoff=backoff,
                                  providers=prov_choices, ratio_cb=ratio_cb, warmup=warmup_toggle))

    df = pd.DataFrame(rows)
    st.write("### Results Summary")
    st.dataframe(df)

    # Quality banner for quantum
    if "quantum" in df["condition"].values:
        qr = float(df.loc[df["condition"]=="quantum","quantum_ratio"].fillna(0).values[0])
        if qr >= min_qratio:
            st.success(f"Quantum source OK (quantum_ratio ≈ {qr:.3f}).")
        elif qr >= 0.5:
            st.warning(f"Quantum source partially degraded (quantum_ratio = {qr:.3f}).")
        else:
            st.error(f"Quantum source heavily degraded (quantum_ratio = {qr:.3f}).")

    # Save artifacts
    out_dir = APP_DIR / "QSW_runs" / (datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_v2.5_streamlit")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)

    # Plots
    p1 = out_dir / "success_rate.png"
    p2 = out_dir / "time_to_escape.png"
    p3 = out_dir / "diversity_bits.png"
    p4 = out_dir / "mi.png"

    def _save(col, title, ylabel, path): save_bar(df, col, title, ylabel, str(path))
    _save("success_rate", "Success Rate — Impasse Escape (v2.5)", "Success Rate", p1)
    _save("avg_time_to_escape", "Avg Time to Escape (lower=better)", "Avg Time to Escape", p2)
    _save("diversity_bits", "Diversity (bits)", "Diversity (bits)", p3)
    _save("early_gate_success_MI", "Early Gate ↔ Success MI (nats)", "MI (nats)", p4)

    st.write("### Plots")
    c1, c2 = st.columns(2); c1.image(str(p1)); c2.image(str(p2))
    c3, c4 = st.columns(2); c3.image(str(p3)); c4.image(str(p4))

    st.write("### Downloads")
    with open(csv_path, "rb") as f:
        st.download_button("results_summary.csv", f, file_name=csv_path.name, mime="text/csv")
