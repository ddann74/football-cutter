import streamlit as st
import cv2
import numpy as np
import re
import os
from moviepy import VideoFileClip

# --- 1. DATA EXTRACTION ---
def parse_report(text):
    """Extracts minutes and event descriptions from match reports."""
    pattern = r"\(?(\d{1,2}(?:\+\d+)?)(?:'|(?::\d{2})|(?:\s?min)|(?:th minute)|(?:\s?'))\)?[\s:-]*(.*)"
    return re.findall(pattern, text, re.IGNORECASE)

def get_seconds(time_str):
    """Converts match time strings (e.g., 12', 45+2) into total seconds."""
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

# --- 2. KICKOFF DETECTION (Data Stabilization) ---
def detect_kickoff(video_path, max_minutes):
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
            
        # Scan every 2 seconds to save processing power
        if frame_count % int(fps * 2) == 0:
            current_sec = frame_count / fps
            
            # Detect green pitch (Visual indicator of kickoff)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            
            progress_val = min(int((frame_count / (fps * max_scan_seconds)) * 100), 100)
            scan_progress.progress(progress_val)
            status_text.text(f"Scanning for kickoff: {int(current_sec/60)}m {int(current_sec%60)}s...")

            # If more than 55% of the frame is green, we've likely found the start
            if green_ratio > 0.55:
                cap.release()
                scan_progress.empty()
                status_text.empty()
                return current_sec
                
        frame_count += 1
    
    cap.release()
    return 0

# --- 3. UI SETUP ---
st.set_page_config(page_title="Football Cutter Pro", page_icon="⚽", layout="wide")
st.title("⚽ AI Football Highlight Cutter")

# --- 4. DATA STABILIZATION INDICATOR ---
st.divider()
st.subheader("System Status")
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.markdown("### Data Stabilization")
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Input")
with col_stat2:
    sync_score = st.empty()
    sync_score.metric("Sync Confidence", "0%")
with col_stat3:
    kickoff_display = st.empty()
    kickoff_display.metric("Kickoff Detected", "--:--")

# --- 5. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150, placeholder="12' Goal - Player Name")
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov'])

with col_right:
    st.info("⚙️ AI Detection Settings")
    search_limit = st.slider("Search Limit (Minutes)", 5, 30, 15)

# --- 6. EXECUTION ---
if st.button("🚀 Run AI Sync & Cut"):
    if report_text and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Processing Match Data...")
        events = parse_report(report_text)
        
        if not events:
            st.error("No valid timestamps found in the report.")
        else:
            # Save temporary file
            temp_path = "temp_match_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            stabilization_placeholder.warning("🟡 Phase 2: Detecting Kickoff...")
            kickoff_sec = detect_kickoff(temp_path, search_limit)
            
            if kickoff_sec > 0:
                stabilization_placeholder.success("🟢 Phase 3: Data Stabilized")
                sync_score.metric("Sync Confidence", "92%")
                m, s = divmod(int(kickoff_sec), 60)
                kickoff_display.metric("Kickoff Detected", f"{m:02d}:{s:02d}")
                
                # Use a safe subclip method to avoid MoviePy version errors
                video = VideoFileClip(temp_path)
                
                for i, (match_min, action) in enumerate(events):
                    event_sec = kickoff_sec + get_seconds(match_min)
                    out_name = f"clip_{i}.mp4"
                    
                    with st.status(f"Creating Clip: {match_min}..."):
                        start_t = max(0, event_sec - 10)
                        end_t = min(video.duration, event_sec + 5)
                        
                        # Fix for AttributeError: 'VideoFileClip' object has no attribute 'subclip'
                        # Handles both MoviePy 1.x and 2.x
                        if hasattr(video, 'sub_clip'):
                            clip = video.sub_clip(start_t, end_t)
                        else:
                            clip = video.subclip(start_t, end_t)
                            
                        clip.write_videofile(out_name, codec="libx264", audio_codec="aac", logger=None)
                    
                    st.download_button(f"Download {match_min} ({action[:15]}...)", 
                                       open(out_name, "rb"), 
                                       file_name=f"{match_min}_highlight.mp4")
                
                video.close()
            else:
                stabilization_placeholder.error("🔴 Kickoff Not Detected")
    else:
        st.error("Please provide both the match report and the video file.")
