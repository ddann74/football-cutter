import streamlit as st
import time

# Styling the app
st.set_page_config(page_title="Football Highlight Cutter", page_icon="⚽")

st.title("⚽ Football Highlight Cutter")

# DATA STABILIZATION SECTION
st.subheader("Data Stabilization")
status_col, metric_col = st.columns([3, 1])

# This is your requested feature!
with status_col:
    status_text = st.empty()
    status_text.write("Status: Waiting for Report")

with metric_col:
    score_val = st.empty()
    score_val.metric("Sync Score", "0%")

# INPUTS
report = st.text_area("1. Paste Match Report", placeholder="12' Goal by Messi...")
video_url = st.text_input("2. Video Link", placeholder="Paste YouTube or MP4 link...")

if st.button("Sync Video & Text"):
    # Simulated Analysis
    status_text.warning("Analyzing Kickoff...")
    score_val.metric("Sync Score", "45%", delta="Yellow")
    time.sleep(2)
    
    status_text.success("Stabilized & Synced")
    score_val.metric("Sync Score", "95%", delta="Green")
    st.balloons()
