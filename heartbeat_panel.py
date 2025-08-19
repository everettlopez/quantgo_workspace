# heartbeat_panel.py — Streamlit panel for R2 heartbeat (no deprecations)
import json, time
import streamlit as st
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError

def _norm_endpoint(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    return raw if raw.startswith("http") else ("https://" + raw) if raw else ""

def _r2_client():
    try:
        endpoint = _norm_endpoint(st.secrets["S3_ENDPOINT_URL"])
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("S3_REGION","auto"),
            config=Config(signature_version="s3v4",
                          s3={"addressing_style": "virtual"},
                          retries={"max_attempts": 3, "mode": "standard"}),
        )
    except Exception as e:
        st.error(f"R2 client init failed: {e}")
        return None

def _get_json(bucket: str, key: str):
    s3 = _r2_client()
    if not s3: return None, "no_client"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8")), None
    except ClientError as ce:
        return None, f"ClientError: {ce.response.get('Error', {}).get('Code', '?')}"
    except BotoCoreError as be:
        return None, f"BotoCoreError: {be}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _list_status_keys(bucket: str, prefix: str):
    s3 = _r2_client()
    if not s3: return None, "no_client"
    try:
        pfx = prefix.rstrip("/") + "/_status/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=pfx, MaxKeys=25)
        return [c["Key"] for c in resp.get("Contents", [])], None
    except Exception as e:
        return None, str(e)

def _fmt_sec(s):
    try:
        s = int(s)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h: return f"{h}h {m}m {s}s"
        if m: return f"{m}m {s}s"
        return f"{s}s"
    except Exception:
        return "—"

def render_heartbeat(title="Collector status (R2 heartbeat)"):
    st.write(f"## {title}")

    # Config
    bucket = st.secrets.get("S3_BUCKET","").strip()
    prefix = st.secrets.get("S3_PREFIX","autocollect").strip().rstrip("/")
    endpoint = _norm_endpoint(st.secrets.get("S3_ENDPOINT_URL",""))

    with st.expander("Connection & path (diagnostics)", expanded=False):
        st.write("**Endpoint:**", f"`{endpoint or '(missing)'}`")
        st.write("**Bucket:**", f"`{bucket or '(missing)'}`")
        st.write("**Prefix:**", f"`{prefix or '(missing)'}`")

    auto = st.checkbox("Auto-refresh (every 10s)", value=True)
    if auto:
        # No deprecation warnings
        st.query_params.update({"_hb": str(int(time.time() // 10))})

    inprog_key = f"{prefix}/_status/in_progress.json" if prefix else "_status/in_progress.json"
    latest_key = f"{prefix}/_status/latest.json"      if prefix else "_status/latest.json"

    c1, c2 = st.columns(2)

    # In-progress pane
    with c1:
        st.subheader("In-progress")
        ip, err_ip = _get_json(bucket, inprog_key)
        if ip:
            phase = ip.get("phase","started")
            cond  = ip.get("condition","—")
            i     = int(ip.get("trial_idx", 0) or 0)
            n     = int(ip.get("n_trials", 0) or 0)
            pr    = float(ip.get("progress", 0) or 0.0)
            el    = ip.get("elapsed_sec","—")
            pure  = ip.get("pure", False)
            provs = ", ".join(ip.get("providers", []) or [])
            qn    = ip.get("q_counts", 0); nq = ip.get("nq_counts", 0); fb = ip.get("fallback_counts", 0)

            st.info(f"Phase **{phase}** — {cond} {i}/{n}  •  Pure: `{pure}`  •  Providers: `{provs}`")
            st.progress(min(max(pr, 0.0), 1.0), text=f"{int(pr*100)}%")
            st.write(f"Elapsed: `{_fmt_sec(el)}`")
            st.write(f"QRNG counts — true:`{qn}`  nonquantum:`{nq}`  fallback:`{fb}`")
            st.code(f"s3://{bucket}/{inprog_key}", language="text")
        else:
            st.write("No in-progress heartbeat found.")
            st.code(f"s3://{bucket}/{inprog_key}", language="text")
            if err_ip: st.caption(f"Fetch error: {err_ip}")

    # Last finished pane
    with c2:
        st.subheader("Last finished")
        lt, err_lt = _get_json(bucket, latest_key)
        if lt:
            run_dir = lt.get("run_dir","—")
            files   = lt.get("files","—")
            qr      = lt.get("quantum_ratio","n/a")
            elfin   = lt.get("elapsed_sec","—")
            ts      = lt.get("ts_utc","—")
            st.success(f"Run **{run_dir}** — Files **{files}** — Quantum ratio **{qr}**")
            st.write(f"Finished in: `{_fmt_sec(elfin)}`  •  Updated (UTC): `{ts}`")
            st.code(f"s3://{bucket}/{prefix}/{run_dir}", language="text")
        else:
            st.write("No finished run recorded yet.")
            st.code(f"s3://{bucket}/{latest_key}", language="text")
            if err_lt: st.caption(f"Fetch error: {err_lt}")

    with st.expander("What files exist under _status/?"):
        keys, err_ls = _list_status_keys(bucket, prefix)
        if keys is not None:
            if keys:
                for k in keys: st.write("•", k)
            else:
                st.write("(no objects found under _status/)")
        else:
            st.caption(f"List error: {err_ls}")

    st.caption("UI build: v2.8 — heartbeat_panel.py")
