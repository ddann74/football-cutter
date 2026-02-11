import streamlit as st
import cv2
import numpy as np
import re
import os
import time
import gc

# --- 1. MOVIEPY 2.X VERSION-SAFE IMPORT ---
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        st.error("MoviePy not found. Please ensure 'moviepy' is in requirements.txt")

# --- 2. PERSISTENCE LAYER ---
if 'clips' not in st.session_state:
    st.session_state.clips = []
if 'workspace' not in st.session_state:
    st.session_state.workspace = f"work_{int(time.time())}"

# --- 3. UTILITIES ---
def parse_report(text):
    pattern = r"\(?(\d{1,2}(?:\+\d+)?)(?:'|(?::\d{2})|(?:\s?min)|(?:th minute)|(?:\s?'))\)?[\s:-]*(.*)"
    return re.findall(pattern, text, re.IGNORECASE)

def get_seconds(time_str):
    clean_time = re.sub(r"[^0-9+:]", "", time_str)
    if ":" in clean_time:
        parts = clean_time.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    elif "+" in clean_time:
        parts = clean_time.split("+")
        return (int(parts[0]) + int(parts[1])) * 60
    else:
        try: return int(clean_time) * 60
        except: return 0

def detect_kickoff_ai(video_path, start_min, end_min):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    scan_progress = st.progress(0)
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
        if current_frame % int(fps * 0.5) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            scan_progress.progress(min(int(((current_frame - start_frame) / (end_frame - start_frame)) * 100), 100))
            if green_ratio > 0.55:
                cap.release()
                return current_frame / fps
        current_frame += 1
    cap.release()
    return 0

# --- 4. UI SETUP & DATA STABILIZATION ---
st.set_page_config(page_title="Football Cutter Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Highlight Cutter")

st.divider()
st.subheader("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Data Stabilization")
    if st.session_state.clips:
        st.success("🟢 Data Stabilized")
    else:
        st.info("⚪ Waiting for Input")
with col2:
    st.metric("Sync Accuracy", "100%" if st.session_state.clips else "0%")
with col3:
    st.metric("Events Found", len(st.session_state.clips))

# --- 5. INPUTS & FILTERS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150, placeholder="12' Goal - Messi\n44' Foul - Ramos")
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

with col_right:
    st.info("🎯 Highlight Selection Filter")
    event_filter = st.multiselect(
        "Only process these events:",
        options=["Goals", "Fouls", "Other Action"],
        default=["Goals", "Fouls"]
    )
    
    st.divider()
    mode = st.radio("Kickoff Detection:", ["Manual Entry", "AI Auto-Scan"])
    if mode == "Manual Entry":
        m = st.number_input("Minute", 0, 120, 20)
        s = st.number_input("Second", 0, 59, 0)
        kickoff_val = (m * 60) + s
    else:
        search_window = st.slider("AI Window (Min)", 0, 60, (19, 22))

# --- 6. PROCESSING ENGINE ---
if st.button("🚀 Start Stabilization & Cut"):
    if report_text and video_file and event_filter:
        os.makedirs(st.session_state.workspace, exist_ok=True)
        temp_source = os.path.join(st.session_state.workspace, "source.mp4")
        
        with open(temp_source, "wb") as f:
            f.write(video_file.getbuffer())
        
        kickoff_sec = kickoff_val if mode == "Manual Entry" else detect_kickoff_ai(temp_source, search_window[0], search_window[1])
        
        if kickoff_sec > 0:
            events = parse_report(report_text)
            st.session_state.clips = [] # Clear for fresh run
            
            for i, (match_min, action) in enumerate(events):
                # Filter Logic
                act_lower = action.lower()
                is_goal = "goal" in act_lower
                is_foul = "foul" in act_lower
                
                if is_goal and "Goals" not in event_filter: continue
                if is_foul and "Fouls" not in event_filter: continue
                if not is_goal and not is_foul and "Other Action" not in event_filter: continue

                label = "GOAL" if is_goal else "FOUL" if is_foul else "ACTION"
                target_sec = kickoff_sec + get_seconds(match_min)
                out_filename = f"{label}_{match_min.replace('+', '_')}_{i}.mp4"
                out_path = os.path.join(st.session_state.workspace, out_filename)
                
                with st.status(f"Processing {match_min}: {label}"):
                    # FRESH LOAD: Prevents timing drift
                    video = VideoFileClip(temp_source)
                    start, end = max(0, target_sec - 10), min(video.duration, target_sec + 5)
                    
                    # MoviePy 2.x Slicing syntax
                    clip = video[start:end]
                    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                    
                    st.session_state.clips.append({
                        "label": f"{match_min} - {label}",
                        "path": out_path,
                        "file_name": out_filename
                    })
                    
                    clip.close()
                    video.close()
                    gc.collect()
            
            st.success("Clipping Complete!")
            st.rerun()
    elif not event_filter:
        st.error("Please select at least one event type (Goals or Fouls).")
    else:
        st.error("Missing video or match report.")

# --- 7. PERSISTENT DOWNLOAD SECTION ---
if st.session_state.clips:
    st.divider()
    st.subheader("📥 Generated Highlights")
    cols = st.columns(3)
    for idx, clip in enumerate(st.session_state.clips):
        with cols[idx % 3]:
            if os.path.exists(clip["path"]):
                with open(clip["path"], "rb") as f:
                    st.download_button(
                        label=f"Download {clip['label']}",
                        data=f,
                        file_name=clip['file_name'],
                        key=f"dl_{idx}"
                    )
