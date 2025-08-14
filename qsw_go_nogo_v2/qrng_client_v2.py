# qrng_client_v2.py — multi-provider QRNG with chunking, retries/backoff, and graceful fallback
# Compatible with your existing app: QRNGClient(...).next() returns float in [0,1].

import os, time, threading, math
import numpy as np
import requests

# ---------------- Provider Interfaces ----------------

class BaseQRNGProvider:
    """Interface for QRNG-like providers. Must return floats in [0,1]."""
    name = "base"
    quantum = True  # set False for non-quantum sources (e.g., NIST Beacon)

    def fetch_uniforms(self, n):
        raise NotImplementedError


class ANUProvider(BaseQRNGProvider):
    """
    ANU QRNG: https://qrng.anu.edu.au/
    JSON schema: {"data":[0..255], ...}
    You can override URL via env QSW_ANU_URL; data key via QSW_ANU_DATA_KEY.
    """
    name = "anu"
    quantum = True

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.url = os.getenv("QSW_ANU_URL", "https://qrng.anu.edu.au/API/jsonI.php")
        self.key = os.getenv("QSW_ANU_DATA_KEY", "data")

    def fetch_uniforms(self, n):
        r = requests.get(self.url, params={"length": int(n), "type": "uint8"}, timeout=self.timeout)
        r.raise_for_status()
        js = r.json() if r.content else {}
        data = js
        # Support nested keys just in case (e.g., "result.data")
        for k in str(self.key).split("."):
            data = data.get(k) if isinstance(data, dict) else None
            if data is None:
                break
        if not data:
            return []
        arr = np.asarray(data, dtype=np.float64) / 255.0
        return arr.clip(0.0, 1.0).tolist()


class NISTBeaconV2Provider(BaseQRNGProvider):
    """
    NIST Randomness Beacon v2 (NOT quantum).
    Endpoint: https://beacon.nist.gov/beacon/2.0/pulse/last
    We convert hex 'outputValue' -> bytes -> [0,1].
    """
    name = "nist"
    quantum = False
    URL = "https://beacon.nist.gov/beacon/2.0/pulse/last"

    def __init__(self, timeout=10):
        self.timeout = timeout
        self._cache_bytes = b""
        self._last_ts = 0.0

    def _pull(self):
        r = requests.get(self.URL, timeout=self.timeout)
        r.raise_for_status()
        js = r.json() if r.content else {}
        pulse = js.get("pulse", {})
        hx = pulse.get("outputValue", "")
        try:
            raw = bytes.fromhex(hx)
        except Exception:
            raw = b""
        self._cache_bytes = raw
        self._last_ts = time.time()

    def fetch_uniforms(self, n):
        # Refresh about once per minute (beacon cadence)
        if not self._cache_bytes or (time.time() - self._last_ts) > 50.0:
            self._pull()
        if not self._cache_bytes:
            return []
        need = int(n)
        rep = math.ceil(need / len(self._cache_bytes)) if self._cache_bytes else 1
        buf = (self._cache_bytes * max(1, rep))[:need]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float64) / 255.0
        return arr.clip(0.0, 1.0).tolist()


class JSONUint8Provider(BaseQRNGProvider):
    """
    Generic JSON provider for services that return an array of 0..255 under a key path.
    Configure via env or constructor:
      - URL:        QSW_JSON_URL
      - DATA KEY:   QSW_JSON_DATA_KEY  (e.g., "data" or "result.data")
      - AUTH ENV:   QSW_JSON_AUTH_ENV  (e.g., "IDQ_API_KEY") — value must also be present in env
      - AUTH HDR:   QSW_JSON_AUTH_HEADER (default "Authorization")
      - AUTH PREF:  QSW_JSON_AUTH_PREFIX (default "Bearer")
      - IS_QUANTUM: QSW_JSON_IS_QUANTUM ("true"/"false")
    """
    name = "json"

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.base_url = os.getenv("QSW_JSON_URL", "")
        self.data_key = os.getenv("QSW_JSON_DATA_KEY", "data")
        auth_env = os.getenv("QSW_JSON_AUTH_ENV", "")
        self.headers = {}
        if auth_env and os.getenv(auth_env):
            hdr = os.getenv("QSW_JSON_AUTH_HEADER", "Authorization")
            pref = os.getenv("QSW_JSON_AUTH_PREFIX", "Bearer")
            self.headers[hdr] = (pref + " " if pref and not pref.endswith(" ") else (pref or "")) + os.getenv(auth_env)
        quantum_flag = os.getenv("QSW_JSON_IS_QUANTUM", "true").strip().lower()
        self.quantum = (quantum_flag != "false")

    def fetch_uniforms(self, n):
        if not self.base_url:
            return []
        params = {"length": int(n)}  # harmless if ignored by the API
        r = requests.get(self.base_url, headers=self.headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        js = r.json() if r.content else {}
        data = js
        for k in str(self.data_key).split("."):
            data = data.get(k) if isinstance(data, dict) else None
            if data is None:
                break
        if not data:
            return []
        arr = np.asarray(data, dtype=np.float64) / 255.0
        return arr.clip(0.0, 1.0).tolist()


# ---------------- Main Client ----------------

class QRNGClient:
    """
    Modes:
      - 'quantum'      : use one or more providers with retries/backoff; top-up with pseudo on shortfall
      - 'pseudo'       : NumPy PCG64
      - 'deterministic': fixed repeating sequence (no true randomness)

    API:
      - next() -> float in [0,1]

    Telemetry counters:
      - true_quantum_count : ints pulled from providers with provider.quantum == True
      - nonquantum_count   : ints pulled from providers with provider.quantum == False (e.g., NIST)
      - fallback_count     : pseudo top-ups when providers fail
      - quantum_ratio()    : true_quantum_count / (true_quantum_count + nonquantum_count + fallback_count)
    """

    def __init__(
        self,
        mode='quantum',
        batch_size=1024,
        max_retries=6,
        backoff=1.6,
        timeout=10,
        providers=None  # list like ["anu","idq","nist"]; if None, read from env QSW_QRNG_PROVIDERS
    ):
        self.mode = mode
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.timeout = int(timeout)

        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0xC0FFEE)

        self.true_quantum_count = 0
        self.nonquantum_count   = 0
        self.fallback_count     = 0

        self._det_seq = np.linspace(0.05, 0.95, 19).tolist()
        self._det_i = 0

        # Build provider chain in priority order
        self._providers = []
        prov_list = providers
        if prov_list is None:
            prov_list = os.getenv("QSW_QRNG_PROVIDERS", "anu").split(",")
        prov_list = [p.strip().lower() for p in prov_list if str(p).strip()]

        for p in prov_list:
            if p == "anu":
                self._providers.append(ANUProvider(timeout=self.timeout))
            elif p in ("idq", "json"):
                # generic JSON-powered provider (configured via env)
                self._providers.append(JSONUint8Provider(timeout=self.timeout))
            elif p == "nist":
                self._providers.append(NISTBeaconV2Provider(timeout=self.timeout))
            # unknown names are ignored

        # Quantum mode uses an internal buffer
        self._buf = [] if self.mode == 'quantum' else None

    # ------------- Public API -------------

    def next(self):
        if self.mode == 'pseudo':
            return float(self._rng.random())
        if self.mode == 'deterministic':
            v = self._det_seq[self._det_i]
            self._det_i = (self._det_i + 1) % len(self._det_seq)
            return float(v)

        # quantum mode
        with self._lock:
            if not self._buf:
                self._refill_chunked(self.batch_size)
            if not self._buf:
                # total miss → single fallback
                self.fallback_count += 1
                return float(self._rng.random())
            return float(self._buf.pop())

    # ------------- Internals -------------

    def _refill_chunked(self, n):
        """Refill buffer by requesting in smaller chunks; try providers in order; top up with pseudo if short."""
        if self.mode != 'quantum':
            return
        want = int(n)
        chunk = min(256, want)   # polite per-request size
        pulled = 0
        while pulled < want:
            need = min(chunk, want - pulled)
            data, is_q = self._fetch_from_any_provider(need)
            if data:
                self._buf.extend(data)
                if is_q:
                    self.true_quantum_count += len(data)
                else:
                    self.nonquantum_count += len(data)
                pulled += len(data)
            else:
                # No provider delivered this round; brief pause then stop loop to avoid spinning
                time.sleep(0.08)
                break

        if pulled < want:
            top_up = want - pulled
            self._buf.extend(self._rng.random(top_up).tolist())
            self.fallback_count += top_up

    def _fetch_from_any_provider(self, n):
        """Try each provider with per-provider retry/backoff. Return (data, is_quantum)."""
        for prov in self._providers:
            delay = 0.25
            for _ in range(self.max_retries + 1):
                try:
                    data = prov.fetch_uniforms(n)
                    if data:
                        return data, getattr(prov, "quantum", True)
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(5.0, delay * self.backoff)
        return [], False

    # ------------- Telemetry -------------

    def quantum_ratio(self):
        q, nq, f = self.true_quantum_count, self.nonquantum_count, self.fallback_count
        tot = q + nq + f
        return (q / tot) if tot > 0 else 0.0
