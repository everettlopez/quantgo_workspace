# report_summary.py — Compact investor-ready PDF for QSW runs (v2.8)
# Reads: results_summary.csv, metadata.json, *.png in the run folder
# Writes: summary.pdf (in the same folder)

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# we use reportlab for a lean, vector PDF
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors


COND_ORDER = ["quantum", "pseudo", "deterministic"]

def _safe_get(d, k, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default

def _draw_title(c, text, y):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75*inch, y, text)
    return y - 0.25*inch

def _draw_subtitle(c, text, y):
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(0.75*inch, y, text)
    return y - 0.18*inch

def _draw_body(c, text, y, size=10.5, leading=13):
    c.setFont("Helvetica", size)
    for line in text.splitlines():
        c.drawString(0.75*inch, y, line)
        y -= (leading/72.0)*inch*72/72  # approximate line height
    return y

def _draw_keyval(c, key, val, y):
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(0.75*inch, y, f"{key}:")
    c.setFont("Helvetica", 10.5)
    c.drawString(2.0*inch, y, str(val))
    return y - 0.18*inch

def _img(c, path: Path, x, y, w):
    try:
        from reportlab.lib.utils import ImageReader
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        h = (w/iw) * ih
        c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, mask='auto')
        return y - h - 0.1*inch
    except Exception:
        return y

def generate_summary_pdf(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    csv_path = run_dir / "results_summary.csv"
    meta_path = run_dir / "metadata.json"
    pdf_path  = run_dir / "summary.pdf"

    # Load data
    df = pd.read_csv(csv_path)
    if "condition" in df.columns:
        df["condition"] = pd.Categorical(df["condition"], COND_ORDER, ordered=True)
        df = df.sort_values("condition")
    meta = json.loads(meta_path.read_text())

    # Pull headline numbers
    def _m(col, cond):
        try:
            return float(df.loc[df["condition"]==cond, col].values[0])
        except Exception:
            return None

    sr_q  = _m("success_rate", "quantum")
    sr_p  = _m("success_rate", "pseudo")
    sr_d  = _m("success_rate", "deterministic")
    tte_q = _m("avg_time_to_escape", "quantum")
    H_q   = _m("diversity_bits", "quantum")
    mi_q  = _m("early_gate_success_MI", "quantum")
    qr_q  = _m("quantum_ratio", "quantum")

    # Headline delta (best available control)
    ctrl_sr = None
    for c in ["pseudo", "deterministic"]:
        v = _m("success_rate", c)
        if v is not None:
            ctrl_sr = v if ctrl_sr is None else max(ctrl_sr, v)
    headline = None
    if sr_q is not None and ctrl_sr is not None and ctrl_sr > 0:
        headline = f"QRNG success rate {'+' if sr_q>=ctrl_sr else ''}{(sr_q-ctrl_sr)*100:.1f} pts vs. best control"
    elif sr_q is not None:
        headline = f"QRNG success rate: {sr_q*100:.1f}%"

    # Build PDF
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER
    y = height - 0.8*inch

    # Cover
    y = _draw_title(c, "Quantum Signal Walker — Run Summary", y)
    c.setFont("Helvetica", 10.5)
    c.setFillColor(colors.gray)
    c.drawString(0.75*inch, y, (run_dir.name))
    c.setFillColor(colors.black)
    y -= 0.22*inch

    y = _draw_subtitle(c, "Headline", y)
    if headline:
        y = _draw_body(c, headline, y)
    else:
        y = _draw_body(c, "Baseline comparison not available yet.", y)

    y -= 0.08*inch
    y = _draw_subtitle(c, "Run Info", y)
    y = _draw_keyval(c, "Timestamp (UTC)", _safe_get(meta, "ts_utc", "—"), y)
    y = _draw_keyval(c, "Version", _safe_get(meta, "app_version", "—"), y)
    y = _draw_keyval(c, "Conditions", ", ".join(_safe_get(meta, "params", {}).get("conditions", [])), y)
    y = _draw_keyval(c, "Pure QRNG", str(_safe_get(meta, "pure_qrng", False)), y)
    y = _draw_keyval(c, "Providers", ", ".join(_safe_get(meta, "providers", [])), y)
    y = _draw_keyval(c, "Trials / cond", _safe_get(meta, "params", {}).get("trials", "—"), y)

    # Page break if needed
    if y < 3.5*inch:
        c.showPage()
        y = height - 0.8*inch

    # Charts grid (2x2)
    y = _draw_subtitle(c, "Key Metrics", y)
    charts = [
        ("success_rate.png",        "Success Rate"),
        ("time_to_escape.png",      "Avg Time to Escape"),
        ("diversity_bits.png",      "Diversity (bits)"),
        ("mi.png",                  "Early Gate ↔ Success MI")
    ]
    col_w = (width - 1.5*inch - 0.75*inch) / 2.0  # left margin + gutter
    x_left = 0.75*inch
    x_right = x_left + col_w + 0.5*inch

    rows = [(0,1), (2,3)]
    for r in rows:
        y_top = y
        for idx, x in zip(r, [x_left, x_right]):
            png = run_dir / charts[idx][0]
            label = charts[idx][1]
            c.setFont("Helvetica", 10.5)
            c.drawString(x, y_top, label)
            y_img = y_top - 0.18*inch
            y_bot = _img(c, png, x, y_img, w=col_w)
        y = min(y_bot, y_top - 2.4*inch)
        if y < 1.3*inch:
            c.showPage()
            y = height - 0.8*inch

    # Takeaway
    c.showPage()
    y = height - 0.8*inch
    y = _draw_subtitle(c, "What this means", y)
    expl = []
    if sr_q is not None:
        expl.append(f"• QRNG success rate: {sr_q*100:.1f}%.")
    if ctrl_sr is not None:
        expl.append(f"• Best control success rate: {ctrl_sr*100:.1f}%.")
    if qr_q is not None:
        expl.append(f"• Quantum ratio during run: {qr_q*100:.1f}% (higher = fewer fallbacks).")
    if H_q is not None:
        expl.append(f"• Diversity under QRNG: {H_q:.2f} bits (higher suggests broader exploration).")
    if mi_q is not None:
        expl.append(f"• Early-gate ↔ success MI (QRNG): {mi_q:.4f} nats (non-zero suggests early randomness relevance).")

    if not expl:
        expl = ["• Run completed. Dataset ready for deeper analysis and replication."]

    y = _draw_body(c, "\n".join(expl), y)

    y -= 0.12*inch
    y = _draw_subtitle(c, "Next suggested step", y)
    y = _draw_body(c, "- Repeat with identical parameters to confirm stability.\n- Then vary one knob (phase length or trials) to probe sensitivity.", y)

    c.save()
    return pdf_path
