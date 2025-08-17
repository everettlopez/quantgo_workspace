# Quantum Signal Walker — Actions Runner (v2.8, single-folder / heartbeat-rich)

#!/usr/bin/env python3
import os, sys, time, json, math, gzip, io, random, requests, threading, queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd

# -------------------------
# Env / config
# -------------------------
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").rstrip("/")
S3_BUCKET       = os.getenv("S3_BUCKET", "")
S3_REGION       = os.getenv("S3_REGION", "auto")
S3_PREFIX       = os.getenv("S3_PREFIX", "").strip().strip("/")

TRIALS          = int(os.getenv("QSW_TRIALS", "3000"))
TRIALS_MODE     = os.getenv("QSW_TRIALS_MODE", "per_condition")  # "per_condition" or "total"
PHASE_LEN       = int(os.getenv("QSW_PHASE_LEN", "8"))           # not used in lite, kept for parity
MAX_STEPS       = int(os.getenv("QSW_MAX_STEPS", "30"))          # limit per trial loop (if used)
CONDITIONS      = [c.strip() for c in os.getenv("QSW_CONDITIONS", "quantum,pseudo,deterministic").split(",") if c.strip()]

PROVIDERS       = [p.strip() for p in os.getenv("QSW_QRNG_PROVIDERS", "anu").split(",") if p.strip()]
PURE_QRNG       = os.getenv("QSW_PURE","1") == "1"
BATCH           = int(os.getenv("QSW_BATCH","4096"))
LOW_WATERMARK   = int(os.getenv("QSW_LOW_WATERMARK", str(max(512, BATCH//2))))
MAX_RETRIES     = int(os.getenv("QSW_MAX_RETRIES","12"))
BACKOFF         = float(os.getenv("QSW_BACKOFF","2.0"))
PURE_MAX_WAIT   = int(os.getenv("QSW_PURE_MAX_WAIT","1500"))   # seconds (capped waiting)
RUN_NAME        = os.getenv("QSW_RUN_NAME","all")

ANU_URL = "https://qrng.anu.edu.au/API/jsonI.php?length={n}&type=uint8"

# -------------------------
# QRNG client (prefetch + strict/pure waiting)
# -------------------------
class QRNGClient:
    def __init__(self, providers: List[str], pure: bool=True, batch_size: int=4096, low_watermark: int=2048,
                 max_retries:int=12, backoff: float=2.0, pure_max_wait:int=1500):
        self.providers     = providers
        self.pure          = pure
        self.batch_size    = batch_size
        self.low_watermark = low_watermark
        self.max_retries   = max_retries
        self.backoff       = backoff
        self.pure_max_wait = pure_max_wait

        self._q = queue.SimpleQueue()
        self._len = 0
        self._lock = threading.Lock()
        self._stop = False
        self.q_counts = 0
        self.fb_counts = 0

        self._t = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._t.start()

    def _fetch_anu(self, n: int) -> List[int]:
        r = requests.get(ANU_URL.format(n=n), timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("data", [])

    def _fetch_once(self, n:int) -> List[float]:
        # Try providers in order (currently just ANU)
        for prov in self.providers:
            try:
                if prov == "anu":
                    data = self._fetch_anu(n)
                else:
                    continue
                if data:
                    return [u/256.0 for u in data]
            except Exception:
                continue
        return []

    def _refill_blocking(self):
        # In pure mode: wait (with capped backoff) until succeed or max wait
        # In non-pure: if fail, we return empty and upstream will fallback to PRNG
        if not self.pure:
            arr = self._fetch_once(self.batch_size)
            if arr:
                self._push(arr)
                return True
            return False

        deadline = time.time() + self.pure_max_wait
        delay = 0.5
        while time.time() < deadline:
            arr = self._fetch_once(self.batch_size)
            if arr:
                self._push(arr)
                return True
            time.sleep(delay)
            delay = min(12.0, delay * self.backoff)
        # give up in pure mode after max wait
        return False

    def _push(self, arr: List[float]):
        for v in arr:
            self._q.put(v)
        with self._lock:
            self._len += len(arr)

    def _prefetch_loop(self):
        while not self._stop:
            need = False
            with self._lock:
                need = (self._len <= self.low_watermark)
            if need:
                ok = self._refill_blocking()
                if not ok and not self.pure:
                    # we'll let caller fallback to PRNG
                    time.sleep(0.05)
            else:
                time.sleep(0.02)

    def next(self) -> float:
        try:
            v = self._q.get_nowait()
            with self._lock:
                self._len -= 1
            self.q_counts += 1
            return v
        except Exception:
            if self.pure:
                # try to refill and block until available or up to max wait per refill call
                ok = self._refill_blocking()
                if not ok:
                    raise RuntimeError("Pure QRNG: providers unavailable; waited to max window.")
                v = self._q.get()
                with self._lock:
                    self._len -= 1
                self.q_counts += 1
                return v
            else:
                # fallback to PRNG
                self.fb_counts += 1
                return random.random()

    def stats(self) -> Dict[str, float]:
        denom = max(1, self.q_counts + self.fb_counts)
        return {
            "q_counts": self.q_counts,
            "fallback_counts": self.fb_counts,
            "quantum_ratio": self.q_counts/denom
        }

    def close(self):
        self._stop = True

# -------------------------
# Simple metrics used in summary
# -------------------------
def shannon_diversity(seq):
    if not seq: return 0.0
    vals, counts = np.unique(seq, return_counts=True)
    p = counts / counts.sum()
    return float(-(p*np.log2(p)).sum())

# Toy “impasse escape” proxy (lightweight but stable across runs)
def run_condition(n_trials:int, condition:str, pure:bool=True) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    qr  = QRNGClient(PROVIDERS, pure=PURE_QRNG, batch_size=BATCH, low_watermark=LOW_WATERMARK,
                     max_retries=MAX_RETRIES, backoff=BACKOFF, pure_max_wait=PURE_MAX_WAIT) if condition=="quantum" else None
    records = []
    for i in range(1, n_trials+1):
        g = (qr.next() if condition=="quantum" else rng.random() if condition=="pseudo" else 0.5)
        steps   = int(10 + (1.0-g)*20)
        success = int(g > 0.5)
        actions = [int(g*10)%3, int(g*100)%2, int(g*1000)%4]
        records.append({"trial": i, "success": success, "time_to_escape": steps, "diversity_bits": shannon_diversity(actions)})
    row = {
        "condition": condition,
        "n_trials": n_trials,
        "success_rate": float(np.mean([r["success"] for r in records])),
        "avg_time_to_escape": float(np.mean([r["time_to_escape"] for r in records])),
        "diversity_bits": float(np.mean([r["diversity_bits"] for r in records])),
        "unique_seq": n_trials,
        "early_gate_success_MI": 0.0,
    }
    if qr:
        row.update(qr.stats())
        qr.close()
    else:
        row.update({"q_counts":0,"fallback_counts":0,"quantum_ratio":0.0})
    return row, records

# -------------------------
# R2 upload helper
# -------------------------
def upload_dir_r2(local_dir: Path):
    import boto3
    from botocore.client import Config
    if not S3_ENDPOINT_URL:
        print("[WARN] No S3 endpoint configured; skipping upload.")
        return 0
    ep = S3_ENDPOINT_URL if S3_ENDPOINT_URL.startswith("http") else ("https://" + S3_ENDPOINT_URL)
    s3 = boto3.client("s3",
        endpoint_url=ep,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=S3_REGION,
        config=Config(signature_version="s3v4", s3={"addressing_style":"virtual"})
    )
    uploaded = 0
    for p in local_dir.rglob("*"):
        if p.is_file():
            key_base = f"{S3_PREFIX}/{local_dir.name}" if S3_PREFIX else local_dir.name
            key = f"{key_base}/{p.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(p), S3_BUCKET, key)
            uploaded += 1
    return uploaded

# -------------------------
# Heartbeat
# -------------------------
def write_heartbeat(out_dir: Path, phase:str, extra:Dict=None):
    status_dir = out_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
               "trials": TRIALS, "conditions": CONDITIONS, "pure": PURE_QRNG,
               "batch": BATCH, "low_watermark": LOW_WATERMARK, "providers": PROVIDERS}
    if extra: payload.update(extra)
    (status_dir/"latest.json").write_text(json.dumps(payload, indent=2))

# -------------------------
# Main
# -------------------------
def main():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"autocollect_{ts}_v2.9_actions_{RUN_NAME}"
    out_dir = Path("QSW_runs")/run_dir_name
    raw_dir = out_dir/"raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # sanity banner
    print(f"[RUN] TRIALS={TRIALS} MODE={TRIALS_MODE} CONDITIONS={CONDITIONS} PURE_QRNG={PURE_QRNG} PROVIDERS={PROVIDERS} BATCH={BATCH} LWM={LOW_WATERMARK}")

    # compute per-condition trials
    nconds = max(1, len(CONDITIONS))
    trials_per_cond = TRIALS if TRIALS_MODE == "per_condition" else max(1, TRIALS // nconds)

    write_heartbeat(out_dir, "started")

    results = []
    # run each condition
    for cond in CONDITIONS:
        print(f"[COND] {cond} → {trials_per_cond} trials")
        try:
            row, recs = run_condition(trials_per_cond, cond, pure=PURE_QRNG)
            results.append(row)
            # write raw
            buf = io.StringIO()
            pd.DataFrame(recs).to_csv(buf, index=False)
            gz = gzip.compress(buf.getvalue().encode())
            (raw_dir/f"{cond}_raw.csv.gz").write_bytes(gz)
        except RuntimeError as e:
            # Pure mode unavailable → record stub and continue
            print(f"[WARN] {cond} failed in pure mode: {e}")
            results.append({
                "condition": cond, "n_trials": 0,
                "success_rate": float("nan"), "avg_time_to_escape": float("nan"),
                "diversity_bits": float("nan"), "unique_seq": 0,
                "early_gate_success_MI": float("nan"),
                "q_counts": 0, "fallback_counts": 0, "quantum_ratio": 0.0,
            })

    # summaries
    df = pd.DataFrame(results)
    df.to_csv(out_dir/"results_summary.csv", index=False)

    meta = {
        "ts_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app_version": "v2.9-actions",
        "params": {"trials": TRIALS, "mode": TRIALS_MODE, "phase_len": PHASE_LEN,
                   "max_steps": MAX_STEPS, "conditions": CONDITIONS},
        "qrng": {"pure": PURE_QRNG, "providers": PROVIDERS,
                 "batch": BATCH, "low_watermark": LOW_WATERMARK,
                 "max_retries": MAX_RETRIES, "backoff": BACKOFF,
                 "pure_max_wait": PURE_MAX_WAIT},
        "run_name": RUN_NAME
    }
    (out_dir/"metadata.json").write_text(json.dumps(meta, indent=2))

    manifest = {
        "folder": out_dir.name,
        "files": [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()]
    }
    (out_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))

    write_heartbeat(out_dir, "finished", {"rows": len(df)})

    # upload to R2
    try:
        n = upload_dir_r2(out_dir)
        print(f"[OK] Uploaded {n} files to r2://{S3_BUCKET}/{S3_PREFIX}/{out_dir.name}")
    except Exception as e:
        print("[WARN] R2 upload failed:", e)

if __name__ == "__main__":
    main()
