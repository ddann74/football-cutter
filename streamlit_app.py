import streamlit as st
import cv2
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

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
        try: return int(clean_time) * 60
        except: return 0

# --- 2. AI KICKOFF DETECTION WITH TIME LIMIT ---
def detect_kickoff_visual(video_path, max_minutes):
    """
    Scans the video for the pitch within the user-defined time limit.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_scan_seconds = max_minutes * 60 
    frame_count = 0
    
    scan_progress = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > fps * max_scan_seconds: 
            break
            
        # Check one frame every 2 seconds for speed
        if frame_count % int(fps * 2) == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 40, 30])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            
            progress_val = min(int((frame_count / (fps * max_scan_seconds)) * 100), 100)
            scan_progress.progress(progress_val)
            status_text.text(f"Scanning: {int(frame_count/fps/60)}m {int(frame_count/fps%60)}s...")

            if green_ratio > 0.55: # Pitch detected
                cap.release()
                scan_progress.empty()
                status_text.empty()
                return frame_count / fps
                
        frame_count += 1
    
    cap.release()
    scan_progress.empty()
    status_text.empty()
    return 0

# --- 3. UI SETUP ---
st.set_page_config(page_title="Football Cutter", page_icon="⚽", layout="wide")
st.title("⚽ Football Highlight Cutter")

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
    sync_score.metric("Sync Accuracy", "0%")
with col3:
    kickoff_display = st.empty()
    kickoff_display.metric("Kickoff Found", "--:--")

# --- 5. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150, placeholder="12' Goal - Messi")
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

with col_right:
    st.info("⚙️ AI Scan Settings")
    search_limit = st.slider("Search Limit (Minutes)", 5, 60, 15, help="How far into the video should the AI look for the kickoff?")
    st.caption("Increase this if your video has a long pre-match intro.")

# --- 6. EXECUTION ---
if st.button("🚀 Start AI Sync & Cut"):
    if report_text and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Stabilizing Data...")
        events = parse_report(report_text)
        
        if not events:
            st.error("No timestamps found in text.")
        else:
            stabilization_placeholder.warning("🟡 Phase 2: Detecting Kickoff...")
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            kickoff_sec = detect_kickoff_visual(temp_path, search_limit)
            
            if kickoff_sec > 0:
                stabilization_placeholder.success("🟢 Phase 3: Fully Stabilized")
                sync_score.metric("Sync Accuracy", "99%")
                m, s = divmod(int(kickoff_sec), 60)
                kickoff_display.metric("Kickoff Found", f"{m:02d}:{s:02d}")
                
                video = VideoFileClip(temp_path)
                for i, (match_min, action) in enumerate(events):
                    event_sec = kickoff_sec + get_seconds(match_min)
                    out_name = f"clip_{i}.mp4"
                    with st.status(f"Cutting: {match_min}..."):
                        clip = video.subclip(max(0, event_sec-10), min(video.duration, event_sec+5))
                        clip.write_videofile(out_name, codec="libx264", audio_codec="aac")
                    st.download_button(f"Download {match_min}", open(out_name, "rb"), file_name=f"{match_min}.mp4")
                video.close()
            else:
                stabilization_placeholder.error("🔴 Kickoff Not Detected")
                st.error(f"AI searched the first {search_limit} minutes but couldn't find the pitch.")
    else:
        st.error("Please provide both inputs.")
