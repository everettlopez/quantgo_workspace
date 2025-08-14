# qrng_client_v2.py — strict quantum-only supported
import os, time, threading, random
from typing import List, Optional
import requests

class QRNGClient:
    """
    Unified QRNG/PRNG/Deterministic interface with strict "quantum_only" support.

    Modes:
      - 'quantum'         → use remote providers (ANU / JSON). If quantum_only=True, never fall back.
      - 'pseudo'          → numpy-like PRNG (here Python random).
      - 'deterministic'   → fixed 0.5 stream.

    Counters (for telemetry):
      - true_quantum_count
      - nonquantum_count
      - fallback_count

    Environment / secrets (optional):
      QSW_QRNG_PROVIDERS="anu,json" (priority order)
      # ANU
      QSW_ANU_URL="https://qrng.anu.edu.au/API/jsonI.php"
      QSW_ANU_DATA_KEY="data"
      # Generic JSON QRNG
      QSW_JSON_URL="https://your-provider/api"
      QSW_JSON_DATA_KEY="data"
      QSW_JSON_AUTH_ENV="IDQ_API_KEY"            # name of another secret containing the token
      QSW_JSON_AUTH_HEADER="Authorization"
      QSW_JSON_AUTH_PREFIX="Bearer"
      QSW_JSON_IS_QUANTUM="true"                 # mark as true quantum in metrics
    """

    def __init__(self,
                 mode: str = 'quantum',
                 batch_size: int = 512,
                 max_retries: int = 6,
                 backoff: float = 1.8,
                 providers: Optional[List[str]] = None):
        self.mode = mode
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self._lock = threading.Lock()
        self._buf: List[float] = []

        # telemetry
        self.true_quantum_count = 0
        self.nonquantum_count = 0
        self.fallback_count = 0

        # provider order
        if providers:
            self.providers = [p.strip() for p in providers if p and p.strip()]
        else:
            env = os.getenv("QSW_QRNG_PROVIDERS", "anu")
            self.providers = [p.strip() for p in env.split(",") if p.strip()]

        # strictness flag; default False; can be set by set_quantum_only(True)
        self._quantum_only = False

        # prebuild headers for json provider if present
        self._json_auth_header = os.getenv("QSW_JSON_AUTH_HEADER", "Authorization")
        self._json_auth_prefix = os.getenv("QSW_JSON_AUTH_PREFIX", "Bearer")
        self._json_auth_env    = os.getenv("QSW_JSON_AUTH_ENV", "")
        self._json_token       = os.getenv(self._json_auth_env, "") if self._json_auth_env else ""
        self._json_is_quantum  = (os.getenv("QSW_JSON_IS_QUANTUM", "true").lower() == "true")

    # --- public toggles ---
    def set_quantum_only(self, flag: bool = True):
        """If True in 'quantum' mode, never fall back to PRNG; retry then raise on failure."""
        self._quantum_only = bool(flag)

    # --- public draw ---
    def next(self) -> float:
        """
        Return a float in [0,1].
        - quantum mode: refill buffer as needed (strict if quantum_only)
        - pseudo: PRNG
        - deterministic: 0.5
        """
        if self.mode == 'quantum':
            with self._lock:
                if not self._buf:
                    self._refill_quantum_strict()
                val = self._buf.pop()
                self.true_quantum_count += 1
                return val
        elif self.mode == 'pseudo':
            self.nonquantum_count += 1
            return random.random()
        else:
            # deterministic 0.5
            self.nonquantum_count += 1
            return 0.5

    # --- internal: strict quantum refill ---
    def _refill_quantum_strict(self):
        """
        Pure mode policy:
          - Try providers in priority order with per-attempt backoff.
          - If none succeed in a cycle, keep cycling providers.
          - If QSW_PURE_MAX_WAIT == 0 → wait indefinitely.
            Else wait up to that many seconds total, then raise.

        Non-pure mode:
          - Fall back to PRNG if all providers fail once.
        """
        # If not in strict mode, do a single pass with retries per provider, then fallback.
        if not self._quantum_only:
            for prov in self.providers:
                delay = 0.3
                for attempt in range(1, self.max_retries + 1):
                    try:
                        if prov == "anu":
                            if self._refill_from_anu(): return
                        elif prov == "json":
                            if self._refill_from_json(): return
                        else:
                            break
                    except Exception:
                        pass
                    time.sleep(delay)
                    delay = min(delay * self.backoff, 8.0)
            # fallback fill with PRNG to keep app responsive
            self._fallback_fill_prng()
            return

        # Strict quantum-only: wait indefinitely or up to a configured cap
        try:
            max_wait_total = float(os.getenv("QSW_PURE_MAX_WAIT", "0"))
        except Exception:
            max_wait_total = 0.0  # 0 means infinite

        start_time = time.time()
        delay_cap = 8.0
        while True:
            for prov in self.providers:
                delay = 0.3
                for attempt in range(1, self.max_retries + 1):
                    try:
                        if prov == "anu":
                            if self._refill_from_anu(): return
                        elif prov == "json":
                            if self._refill_from_json(): return
                        else:
                            break
                    except Exception:
                        pass
                    time.sleep(min(delay, delay_cap))
                    delay = min(delay * self.backoff, delay_cap)

            if max_wait_total > 0 and (time.time() - start_time) >= max_wait_total:
                raise RuntimeError(
                    "Pure QRNG is enabled: providers unavailable; giving up without fallback."
                )

    def _fallback_fill_prng(self):
        self._buf = [random.random() for _ in range(self.batch_size)]
        self.fallback_count += len(self._buf)

    # --- provider fetchers ---
    def _refill_from_anu(self) -> bool:
        url = os.getenv("QSW_ANU_URL", "https://qrng.anu.edu.au/API/jsonI.php")
        n = self.batch_size
        # 'length' can be up to 1024; fetch in chunks if large
        remaining = n
        accum: List[float] = []
        while remaining > 0:
            take = min(remaining, 1024)
            r = requests.get(f"{url}?length={take}&type=uint8", timeout=15)
            r.raise_for_status()
            js = r.json()
            data_key = os.getenv("QSW_ANU_DATA_KEY", "data")
            raw = js.get(data_key, [])
            if not raw or not isinstance(raw, list):
                raise ValueError("ANU response missing data.")
            # scale 0..255 → [0,1]
            accum.extend([min(max(x / 255.0, 0.0), 1.0) for x in raw])
            remaining -= take
        if not accum:
            return False
        # push onto buffer (pop() friendly)
        self._buf.extend(accum)
        return True

    def _refill_from_json(self) -> bool:
        url = os.getenv("QSW_JSON_URL", "")
        if not url:
            raise ValueError("QSW_JSON_URL not set")
        n = self.batch_size
        headers = {}
        if self._json_token:
            headers[self._json_auth_header] = f"{self._json_auth_prefix} {self._json_token}".strip()
        remaining = n
        accum: List[float] = []
        while remaining > 0:
            take = min(remaining, 1024)
            r = requests.get(f"{url}?length={take}", headers=headers, timeout=15)
            r.raise_for_status()
            js = r.json()
            key = os.getenv("QSW_JSON_DATA_KEY", "data")
            # support nested key like "result.data"
            data = js
            for part in key.split("."):
                data = data.get(part, None) if isinstance(data, dict) else None
            if not data or not isinstance(data, list):
                raise ValueError("JSON QRNG response missing data list")
            accum.extend([min(max(int(x) / 255.0, 0.0), 1.0) for x in data])
            remaining -= take
        if not accum:
            return False
        self._buf.extend(accum)
        return True
