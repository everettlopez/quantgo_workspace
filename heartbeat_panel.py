# heartbeat_panel.py — tiny Streamlit panel to show collector status from R2
import json, time, streamlit as st
import boto3
from botocore.client import Config

def _r2_client():
    try:
        return boto3.client(
            "s3",
            endpoint_url=st.secrets["S3_ENDPOINT_URL"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("S3_REGION","auto"),
            config=Config(signature_version="s3v4", s3={"addressing_style":"path"}),
        )
    except Exception:
        return None

def _get_json(bucket: str, key: str):
    s3 = _r2_client()
    if not s3: return None
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

def render_heartbeat(title="Collector status"):
    st.write(f"## {title}")
    bucket = st.secrets.get("S3_BUCKET", "")
    prefix = st.secrets.get("S3_PREFIX", "autocollect").rstrip("/")
    inprog_key = f"{prefix}/_status/in_progress.json"
    latest_key = f"{prefix}/_status/latest.json"

    auto = st.checkbox("Auto-refresh (10s)", value=True)
    if auto:
        st.experimental_set_query_params(_=int(time.time() // 10))

    col1, col2 = st.columns(2)
    with col1:
        st.caption("In-progress")
        ip = _get_json(bucket, inprog_key)
        if ip:
            st.info(f"Phase **{ip.get('phase','?')}** — {ip.get('condition','?')} "
                    f"{ip.get('trial_idx','?')}/{ip.get('n_trials','?')}")
            st.write(f"Updated (UTC): `{ip.get('ts_utc','?')}`")
        else:
            st.write("No in-progress heartbeat.")

    with col2:
        st.caption("Last finished")
        lt = _get_json(bucket, latest_key)
        if lt:
            st.success(f"Run **{lt.get('run_dir','?')}** — Files **{lt.get('files','?')}**")
            st.write(f"Quantum ratio: **{lt.get('quantum_ratio','n/a')}**")
            st.write(f"Updated (UTC): `{lt.get('ts_utc','?')}`")
            st.code(f"s3://{bucket}/{prefix}/{lt.get('run_dir','')}", language="text")
        else:
            st.write("No finished run recorded yet.")
