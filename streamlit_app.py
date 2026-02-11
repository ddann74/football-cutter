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

# --- 2. SESSION STATE INITIALIZATION ---
# This prevents data loss when downloading
if 'processed_clips' not in st.session_state:
    st.session_state.processed_clips = []
if 'workspace' not in st.session_state:
    st.session_state.workspace = None

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

# --- 4. UI SETUP ---
st.set_page_config(page_title="Football Cutter Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Highlight Cutter")

st.divider()
st.subheader("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Data Stabilization")
    stab_val = st.empty()
    if st.session_state.processed_clips:
        stab_val.success("🟢 Data Stabilized")
    else:
        stab_val.info("⚪ Waiting for Input")

# --- 5. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150)
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

with col_right:
    mode = st.radio("Detection Method:", ["Manual Entry", "AI Auto-Scan"])
    if mode == "Manual Entry":
        man_min = st.number_input("Kickoff Minute", 0, 120, 20)
        man_sec = st.number_input("Kickoff Second", 0, 59, 0)
        kickoff_sec_input = (man_min * 60) + man_sec
    else:
        search_window = st.slider("AI Search Window (Min)", 0, 60, (19, 22))

# --- 6. PROCESSING ENGINE ---
if st.button("🚀 Process Clips"):
    if report_text and video_file:
        # Create a persistent workspace for this session
        session_id = str(int(time.time()))
        st.session_state.workspace = f"work_{session_id}"
        os.makedirs(st.session_state.workspace, exist_ok=True)
        
        temp_source = os.path.join(st.session_state.workspace, "source.mp4")
        with open(temp_source, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Calculate Kickoff
        if mode == "Manual Entry":
            kickoff_sec = kickoff_sec_input
        else:
            kickoff_sec = detect_kickoff_ai(temp_source, search_window[0], search_window[1])
        
        if kickoff_sec > 0:
            events = parse_report(report_text)
            temp_results = []
            
            for i, (match_min, action) in enumerate(events):
                target_sec = kickoff_sec + get_seconds(match_min)
                label = "GOAL" if "goal" in action.lower() else "FOUL" if "foul" in action.lower() else "ACTION"
                out_path = os.path.join(st.session_state.workspace, f"{label}_{i}.mp4")
                
                with st.status(f"Cutting {match_min}..."):
                    video = VideoFileClip(temp_source)
                    start, end = max(0, target_sec - 10), min(video.duration, target_sec + 5)
                    clip = video[start:end] # MoviePy 2.x slicing
                    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
                    clip.close()
                    video.close()
                
                temp_results.append({
                    "label": f"{match_min} - {label}",
                    "path": out_path,
                    "filename": f"{label}_{match_min}.mp4"
                })
            
            # Save to session state so they persist across reruns
            st.session_state.processed_clips = temp_results
            st.rerun()

# --- 7. PERSISTENT DOWNLOAD SECTION ---
if st.session_state.processed_clips:
    st.divider()
    st.subheader("📥 Your Highlights")
    # Display clips in a grid
    cols = st.columns(3)
    for idx, clip_data in enumerate(st.session_state.processed_clips):
        with cols[idx % 3]:
            if os.path.exists(clip_data["path"]):
                with open(clip_data["path"], "rb") as f:
                    st.download_button(
                        label=clip_data["label"],
                        data=f,
                        file_name=clip_data["filename"],
                        key=f"btn_{idx}" # Unique key is vital
                    )
            else:
                st.error("File lost. Please re-process.")

if st.button("🗑️ Clear Workspace"):
    st.session_state.processed_clips = []
    st.rerun()
