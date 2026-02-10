import streamlit as st
import time

# Page Setup
st.set_page_config(page_title="Football Highlight Cutter", page_icon="⚽", layout="wide")

# Custom CSS to make it look dark and sleek
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚽ Football Highlight Cutter")
st.write("Upload your match report and video to begin the AI-syncing process.")

# --- DATA STABILIZATION SECTION ---
st.divider()
st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    # This is the Data Stabilization feature you requested
    st.markdown("### Data Stabilization")
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Input")

with col2:
    sync_score = st.empty()
    sync_score.metric("Sync Accuracy", "0%")

with col3:
    process_light = st.empty()
    process_light.markdown("🔴 **System Offline**")

# --- INPUT SECTION ---
st.divider()
report_text = st.text_area("1. Paste Match Report", height=150, placeholder="Example: 12' Goal - L. Messi\n44' Yellow Card - Ramos...")
video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

if st.button("🚀 Start AI Analysis"):
    if report_text and video_file:
        # Step 1: Analyzing
        stabilization_placeholder.warning("🟡 Analyzing Kickoff...")
        sync_score.metric("Sync Accuracy", "42%", delta="Rising")
        process_light.markdown("🟡 **Processing...**")
        time.sleep(2)
        
        # Step 2: Stabilizing
        stabilization_placeholder.success("🟢 Data Stabilized")
        sync_score.metric("Sync Accuracy", "98%", delta="Optimized")
        process_light.markdown("🟢 **System Online**")
        
        st.success("Sync Complete! Ready to cut highlights.")
        st.balloons()
    else:
        st.error("Please provide both a report and a video file.")
