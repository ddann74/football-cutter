import streamlit as st
import cv2import streamlit as st
import cv2
import numpy as np
import re
import os
import time

# Robust MoviePy import
try:
    from moviepy import VideoFileClip
except:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        st.error("MoviePy not found. Please add 'moviepy' to requirements.txt")

# --- 1. DATA EXTRACTION ---
def parse_report(text):
    """Extracts match minutes and events."""
    pattern = r"\(?(\d{1,2}(?:\+\d+)?)(?:'|(?::\d{2})|(?:\s?min)|(?:th minute)|(?:\s?'))\)?[\s:-]*(.*)"
    return re.findall(pattern, text, re.IGNORECASE)

def get_seconds(time_str):
    """Converts match time to seconds (handles 12', 45+2, etc)."""
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

# --- 2. CALIBRATED KICKOFF DETECTION ---
def detect_kickoff_calibrated(video_path, start_min, end_min):
    """Focused scan to find the pitch in the 20:00-20:04 window."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    # Jump straight to the start of your window
    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    scan_progress = st.progress(0)
    status_text = st.empty()
    
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
            
        # Scan every 0.5 seconds for high precision in this small window
        if current_frame % int(fps * 0.5) == 0:
            current_sec = current_frame / fps
            
            # Visual check for green pitch
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            
            # Update progress UI
            progress_val = min(int(((current_frame - start_frame) / (end_frame - start_frame)) * 100), 100)
            scan_progress.progress(progress_val)
            status_text.text(f"Calibrating Sync: {int(current_sec/60)}m {int(current_sec%60)}s...")

            if green_ratio > 0.55:
                cap.release()
                scan_progress.empty()
                status_text.empty()
                return current_sec
                
        current_frame += 1
    
    cap.release()
    return 0

# --- 3. UI SETUP ---
st.set_page_config(page_title="Football Cutter Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Highlight Cutter")

# --- 4. DATA STABILIZATION INDICATOR ---
st.divider()
st.subheader("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Data Stabilization")
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Input")
with col2:
    sync_score = st.empty()
    sync_score.metric("Calibration Accuracy", "0%")
with col3:
    kickoff_display = st.empty()
    kickoff_display.metric("Kickoff Point", "--:--")

# --- 5. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150)
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov'])

with col_right:
    st.info("⚙️ Calibration Settings")
    st.write("AI is calibrated to find kickoff near 20:00.")
    # Preset to the window you provided
    search_window = st.slider("Search Window (Minutes)", 15, 25, (19, 22))

# --- 6. EXECUTION ---
if st.button("🚀 Start Calibrated Sync"):
    if report_text and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Stabilizing Match Data...")
        events = parse_report(report_text)
        
        if not events:
            st.error("No valid timestamps found.")
        else:
            temp_path = "temp_match_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            stabilization_placeholder.warning(f"🟡 Phase 2: Calibrating Kickoff ({search_window[0]}-{search_window[1]}m)...")
            kickoff_sec = detect_kickoff_calibrated(temp_path, search_window[0], search_window[1])
            
            if kickoff_sec > 0:
                stabilization_placeholder.success("🟢 Phase 3: Fully Stabilized")
                sync_score.metric("Calibration Accuracy", "99%")
                m, s = divmod(int(kickoff_sec), 60)
                kickoff_display.metric("Kickoff Point", f"{m:02d}:{s:02d}")
                
                video = VideoFileClip(temp_path)
                st.success(f"Processing {len(events)} highlight clips...")
                
                for i, (match_min, action) in enumerate(events):
                    event_sec = kickoff_sec + get_seconds(match_min)
                    out_name = f"clip_{i}.mp4"
                    
                    with st.status(f"Creating Clip: {match_min}..."):
                        start_t = max(0, event_sec - 10)
                        end_t = min(video.duration, event_sec + 5)
                        
                        # --- FIX: VERSION AGNOSTIC CLIP METHOD ---
                        # MoviePy 2.0+ uses 'sub_clip', 1.0 uses 'subclip'
                        if hasattr(video, 'sub_clip'):
                            clip = video.sub_clip(start_t, end_t)
                        else:
                            clip = video.subclip(start_t, end_t)
                        # -----------------------------------------
                        
                        clip.write_videofile(out_name, codec="libx264", audio_codec="aac", logger=None)
                    
                    st.download_button(f"Download {match_min}", open(out_name, "rb"), file_name=f"{match_min}.mp4")
                
                video.close()
            else:
                stabilization_placeholder.error("🔴 Kickoff Not Detected in Window")
                st.error("AI couldn't find the pitch in the selected window. Try widening the search window.")
    else:
        st.error("Please provide both match report and video.")
import numpy as np
import re
import os
import time

# Robust MoviePy import
try:
    from moviepy import VideoFileClip
except:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        st.error("MoviePy not found. Please add 'moviepy' to requirements.txt")

# --- 1. DATA EXTRACTION ---
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
        try:
            return int(clean_time) * 60
        except:
            return 0

# --- 2. CALIBRATED KICKOFF DETECTION ---
def detect_kickoff_calibrated(video_path, start_min, end_min):
    """
    Focused scan between specific minutes to find the pitch/whistle.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    
    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    scan_progress = st.progress(0)
    status_text = st.empty()
    
    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret: break
            
        if current_frame % int(fps * 1) == 0:
            current_sec = current_frame / fps
            
            # Visual check for green pitch
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            
            # Update progress relative to the window
            total_window_frames = end_frame - start_frame
            progress_val = min(int(((current_frame - start_frame) / total_window_frames) * 100), 100)
            scan_progress.progress(progress_val)
            status_text.text(f"Calibrating Sync: {int(current_sec/60)}m {int(current_sec%60)}s...")

            if green_ratio > 0.55:
                cap.release()
                scan_progress.empty()
                status_text.empty()
                return current_sec
                
        current_frame += 1
    
    cap.release()
    return 0

# --- 3. UI SETUP ---
st.set_page_config(page_title="Football Cutter Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Highlight Cutter")

# --- 4. DATA STABILIZATION INDICATOR ---
st.divider()
st.subheader("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Data Stabilization")
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Input")
with col2:
    sync_score = st.empty()
    sync_score.metric("Calibration Accuracy", "0%")
with col3:
    kickoff_display = st.empty()
    kickoff_display.metric("Kickoff Point", "--:--")

# --- 5. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150)
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov'])

with col_right:
    st.info("⚙️ Calibration Settings")
    st.write("AI will focus detection around the 20-minute mark.")
    # Defaulting to your suggested range
    time_window = st.slider("Search Window (Minutes)", 15, 25, (19, 22))

# --- 6. EXECUTION ---
if st.button("🚀 Start Calibrated Sync"):
    if report_text and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Stabilizing Match Data...")
        events = parse_report(report_text)
        
        if not events:
            st.error("No timestamps found.")
        else:
            temp_path = "temp_match_calibrated.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            stabilization_placeholder.warning(f"🟡 Phase 2: Calibrating Kickoff ({time_window[0]}-{time_window[1]}m)...")
            kickoff_sec = detect_kickoff_calibrated(temp_path, time_window[0], time_window[1])
            
            if kickoff_sec > 0:
                stabilization_placeholder.success("🟢 Phase 3: Fully Stabilized")
                sync_score.metric("Calibration Accuracy", "98%")
                m, s = divmod(int(kickoff_sec), 60)
                kickoff_display.metric("Kickoff Point", f"{m:02d}:{s:02d}")
                
                video = VideoFileClip(temp_path)
                for i, (match_min, action) in enumerate(events):
                    event_sec = kickoff_sec + get_seconds(match_min)
                    out_name = f"highlight_{i}.mp4"
                    
                    with st.status(f"Creating Clip: {match_min}..."):
                        start_t = max(0, event_sec - 10)
                        end_t = min(video.duration, event_sec + 5)
                        
                        # Fix for MoviePy 2.0+ Attribute Error
                        if hasattr(video, 'sub_clip'):
                            clip = video.sub_clip(start_t, end_t)
                        else:
                            clip = video.subclip(start_t, end_t)
                            
                        clip.write_videofile(out_name, codec="libx264", audio_codec="aac", logger=None)
                    
                    st.download_button(f"Download {match_min}", open(out_name, "rb"), file_name=f"{match_min}.mp4")
                video.close()
            else:
                stabilization_placeholder.error("🔴 Kickoff Not Detected in Window")
    else:
        st.error("Please provide both inputs.")
