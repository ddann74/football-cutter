import streamlit as st
import time
import re
import os
from moviepy.editor import VideoFileClip # The Scissors

# --- LOGIC FUNCTIONS ---
def parse_report(text):
    pattern = r"(\d{1,2}\+?\d?[':])\s*(.*)"
    return re.findall(pattern, text)

def get_seconds(time_str):
    """Converts 00:42 or 12' to total seconds."""
    time_str = time_str.replace("'", "")
    if ":" in time_str:
        m, s = map(int, time_str.split(":"))
        return m * 60 + s
    return int(time_str) * 60

# --- PAGE SETUP ---
st.set_page_config(page_title="Football Highlight Cutter", page_icon="⚽", layout="wide")
st.title("⚽ Football Highlight Cutter")

# --- INPUT SECTION ---
report_text = st.text_area("1. Paste Match Report", height=150)
video_file = st.file_uploader("2. Upload Match Video", type=['mp4'])

if st.button("🚀 Cut Highlights"):
    if report_text and video_file:
        # 1. Save uploaded file
        with open("raw_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        # 2. Data Stabilization (Parsing)
        events = parse_report(report_text)
        kickoff_sec = 42  # Simplified: Assuming kickoff is at 42 seconds
        
        st.write(f"### ✂️ Cutting {len(events)} Highlights...")
        
        video = VideoFileClip("raw_video.mp4")
        
        for i, (match_min, action) in enumerate(events):
            # Calculate timing: Start 10s before event, end 5s after
            event_sec = kickoff_sec + get_seconds(match_min)
            start_time = max(0, event_sec - 10)
            end_time = min(video.duration, event_sec + 5)
            
            st.write(f"Processing: {match_min} - {action}...")
            
            # Cut the clip
            new_clip = video.subclip(start_time, end_time)
            output_filename = f"highlight_{i}.mp4"
            new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
            
            # Provide Download Button
            with open(output_filename, "rb") as file:
                st.download_button(label=f"Download {action}", data=file, file_name=output_filename)
        
        video.close()
        st.success("All highlights cut successfully!")
    else:
        st.error("Missing input data.")
