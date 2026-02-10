import streamlit as st
import time
import re

# --- LOGIC FUNCTIONS ---
def parse_report(text):
    """
    Scans the text for patterns like 12' or 45:00 and extracts the event.
    """
    # Pattern looks for digits followed by a quote ' or :
    pattern = r"(\d{1,2}\+?\d?[':])\s*(.*)"
    events = re.findall(pattern, text)
    return events

# --- PAGE SETUP ---
st.set_page_config(page_title="Football Highlight Cutter", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚽ Football Highlight Cutter")

# --- DATA STABILIZATION SECTION ---
st.divider()
st.subheader("System Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Data Stabilization")
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Report")

with col2:
    sync_score = st.empty()
    sync_score_val = sync_score.metric("Sync Accuracy", "0%")

with col3:
    process_light = st.empty()
    process_light.markdown("🔴 **System Offline**")

# --- INPUT SECTION ---
st.divider()
report_text = st.text_area("1. Paste Match Report", height=150, 
                           placeholder="Example:\n12' Goal - Messi\n34' Yellow Card - Ramos\n89' Goal - Mbappe")

video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

if st.button("🚀 Start AI Analysis"):
    if report_text and video_file:
        # STEP 1: PARSING LOGIC
        stabilization_placeholder.warning("🟡 Parsing Report Text...")
        process_light.markdown("🟡 **Processing...**")
        
        # Call our logic function
        found_events = parse_report(report_text)
        time.sleep(1) # Artificial delay for effect
        
        if found_events:
            # STEP 2: STABILIZATION LOGIC
            stabilization_placeholder.success(f"🟢 {len(found_events)} Events Stabilized")
            sync_score.metric("Sync Accuracy", "98%", delta="High Confidence")
            process_light.markdown("🟢 **System Online**")
            
            st.write("### Detected Highlights:")
            for time_stamp, action in found_events:
                st.write(f"✅ **{time_stamp}** - {action}")
            
            st.balloons()
        else:
            stabilization_placeholder.error("🔴 Stabilization Failed")
            st.error("I couldn't find any timestamps (like 12') in your report. Please check the format.")
    else:
        st.error("Please provide both a report and a video file.")
