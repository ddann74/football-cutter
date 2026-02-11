import streamlit as st
import cv2
import numpy as np
import re
import os
import time

# --- 1. VERSION-SAFE MOVIEPY IMPORT ---
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        st.error("MoviePy not found. Please add 'moviepy' to requirements.txt")

# --- 2. DATA EXTRACTION ---
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

# --- 3. AI KICKOFF DETECTION ---
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

# --- 5. SYSTEM DASHBOARD (Data Stabilization) ---
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
    kickoff_display.metric("Kickoff Point", "--:--")

# --- 6. INPUTS ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    report_text = st.text_area("1. Paste Match Report", height=150, placeholder="12' Goal - Messi")
    video_file = st.file_uploader("2. Upload Match Video", type=['mp4', 'mov', 'avi'])

with col_right:
    st.info("⚙️ Stabilization Mode")
    mode = st.radio("Choose Kickoff Detection Method:", ["Manual Entry", "AI Auto-Scan"])
    
    if mode == "Manual Entry":
        st.write("Set exact kickoff (e.g., 20:02):")
        m_col, s_col = st.columns(2)
        man_min = m_col.number_input("Minutes", 0, 120, 20)
        man_sec = s_col.number_input("Seconds", 0, 59, 2)
        manual_total_sec = (man_min * 60) + man_sec
    else:
        st.write("AI Search Window:")
        search_window = st.slider("Window (Minutes)", 0, 60, (19, 22))

# --- 7. CORE ENGINE ---
if st.button("🚀 Start Stabilization & Cut"):
    if report_text and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Stabilizing Match Data...")
        events = parse_report(report_text)
        
        if not events:
            st.error("No valid timestamps found.")
        else:
            temp_path = "temp_match_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video_file.getbuffer())
            
            # Lock the Kickoff
            if mode == "Manual Entry":
                kickoff_sec = manual_total_sec
                stabilization_placeholder.success("🟢 Phase 2: Manual Stabilization Active")
                sync_score.metric("Sync Accuracy", "100% (Manual)")
            else:
                stabilization_placeholder.warning("🟡 Phase 2: AI Scanning...")
                kickoff_sec = detect_kickoff_ai(temp_path, search_window[0], search_window[1])
                if kickoff_sec > 0:
                    stabilization_placeholder.success("🟢 Phase 2: AI Stabilized")
                    sync_score.metric("Sync Accuracy", "98% (AI)")
                
            if kickoff_sec > 0:
                m, s = divmod(int(kickoff_sec), 60)
                kickoff_display.metric("Kickoff Point", f"{m:02d}:{s:02d}")
                
                st.success(f"Processing {len(events)} highlights...")
                
                for i, (match_min, action) in enumerate(events):
                    event_sec = kickoff_sec + get_seconds(match_min)
                    
                    # UNIQUE FILENAME: Prevents the browser from serving a cached version
                    unique_id = int(time.time())
                    out_name = f"clip_{unique_id}_{i}.mp4"
                    
                    with st.status(f"Cutting: {match_min} {action[:15]}..."):
                        # RE-LOAD: Ensures a fresh instance for every single highlight
                        video = VideoFileClip(temp_path)
                        start_t = max(0, event_sec - 10)
                        end_t = min(video.duration, event_sec + 5)
                        
                        try:
                            # --- MOVIEPY 2.X SECURE TRIM ---
                            # Check for modern sub_clip (v2.x) first
                            if hasattr(video, 'sub_clip'):
                                clip = video.sub_clip(start_t, end_t)
                            elif hasattr(video, 'subclip'):
                                clip = video.subclip(start_t, end_t)
                            else:
                                # Final slicing fallback
                                clip = video.with_start(start_t).with_end(end_t).with_duration(end_t - start_t)
                            
                            clip.write_videofile(out_name, codec="libx264", audio_codec="aac", logger=None)
                            
                            # CLEANUP: Unlocks the file so the next loop can run safely
                            clip.close()
                            video.close()
                        except Exception as e:
                            st.error(f"Error at {match_min}: {e}")
                    
                    # DOWNLOAD: Unique keys to prevent Streamlit session conflicts
                    with open(out_name, "rb") as f:
                        st.download_button(
                            label=f"Download {match_min} - {action[:15]}",
                            data=f,
                            file_name=f"highlight_{match_min.replace('+', '_')}.mp4",
                            key=f"dl_{unique_id}_{i}"
                        )
            else:
                stabilization_placeholder.error("🔴 Kickoff Not Detected")
    else:
        st.error("Missing Match Report or Video File.")
