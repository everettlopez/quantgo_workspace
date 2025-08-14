# heartbeat_panel.py — R2 heartbeat panel with built-in diagnostics
import json, time, traceback
import streamlit as st

# You need boto3 in your Streamlit app; if missing add "boto3" to requirements.txt.
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

def _norm_endpoint(raw: str) -> str:
    if not raw: return ""
    ep = raw.strip().rstrip("/")
    if not ep.startswith("http"):
        ep = "https://" + ep
    return ep

def _r2_client():
    try:
        endpoint = _norm_endpoint(st.secrets["S3_ENDPOINT_URL"])
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("S3_REGION", "auto"),
            # R2 generally prefers virtual-hosted addressing, v4 signing
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
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data), None
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
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=pfx, MaxKeys=10)
        keys = [c["Key"] for c in resp.get("Contents", [])]
        return keys, None
    except Exception as e:
        return None, str(e)

def render_heartbeat(title="Collector status (R2 heartbeat)"):
    st.write(f"## {title}")

    # Read config from secrets
    bucket = st.secrets.get("S3_BUCKET", "").strip()
    prefix = st.secrets.get("S3_PREFIX", "autocollect").strip().rstrip("/")
    endpoint = _norm_endpoint(st.secrets.get("S3_ENDPOINT_URL", ""))

    with st.expander("Connection & path (diagnostics)", expanded=False):
        st.write("**Endpoint:**", f"`{endpoint or '(missing)'}`")
        st.write("**Bucket:**", f"`{bucket or '(missing)'}`")
        st.write("**Prefix:**", f"`{prefix or '(missing)'}`")
        st.caption("These must match the runner’s env/secrets exactly.")

    auto = st.checkbox("Auto-refresh (every 10s)", value=True)
    if auto:
        st.experimental_set_query_params(_=int(time.time() // 10))

    inprog_key = f"{prefix}/_status/in_progress.json" if prefix else "_status/in_progress.json"
    latest_key = f"{prefix}/_status/latest.json"      if prefix else "_status/latest.json"

    cols = st.columns(2)
    with cols[0]:
        st.subheader("In-progress")
        ip, err_ip = _get_json(bucket, inprog_key)
        if ip:
            st.info(f"Phase **{ip.get('phase','?')}** — {ip.get('condition','?')} "
                    f"{ip.get('trial_idx','?')}/{ip.get('n_trials','?')}")
            st.write(f"Updated (UTC): `{ip.get('ts_utc','?')}`")
        else:
            st.write("No in-progress heartbeat found.")
            st.code(f"s3://{bucket}/{inprog_key}", language="text")
            if err_ip: st.caption(f"Fetch error: {err_ip}")

    with cols[1]:
        st.subheader("Last finished")
        lt, err_lt = _get_json(bucket, latest_key)
        if lt:
            st.success(f"Run **{lt.get('run_dir','?')}** — Files **{lt.get('files','?')}**")
            st.write(f"Quantum ratio: **{lt.get('quantum_ratio','n/a')}**")
            st.write(f"Updated (UTC): `{lt.get('ts_utc','?')}`")
            st.code(f"s3://{bucket}/{prefix}/{lt.get('run_dir','')}", language="text")
        else:
            st.write("No finished run recorded yet.")
            st.code(f"s3://{bucket}/{latest_key}", language="text")
            if err_lt: st.caption(f"Fetch error: {err_lt}")

    with st.expander("What files exist under _status/?"):
        keys, err_ls = _list_status_keys(bucket, prefix)
        if keys is not None:
            if keys:
                for k in keys:
                    st.write("•", k)
            else:
                st.write("(no objects found under _status/)")
        else:
            st.caption(f"List error: {err_ls}")

    st.caption("Tip: If keys aren’t found, check that your runner and app use the **same** bucket and prefix.")
