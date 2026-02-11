import streamlit as st
import cv2
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from moviepy.video.io.VideoFileClip import VideoFileClip

# --- 1. NEW LOGIC: WEB SCRAPER ---
def scrape_report_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # This gets all text from the page; our regex will filter the timestamps
        return soup.get_text()
    except Exception as e:
        st.error(f"Could not read URL: {e}")
        return ""

# --- 2. EXISTING LOGIC ---
def parse_report(text):
    pattern = r"(\d{1,2}\+?\d?[':])\s*(.*)"
    return re.findall(pattern, text)

def get_seconds(time_str):
    time_str = time_str.replace("'", "")
    if ":" in time_str:
        m, s = map(int, time_str.split(":"))
        return m * 60 + s
    return int(time_str) * 60

# --- 3. UI SETUP ---
st.set_page_config(page_title="Football Highlight Cutter", page_icon="⚽", layout="wide")
st.title("⚽ Football Highlight Cutter")

# --- 4. DATA STABILIZATION INDICATOR ---
st.subheader("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    stabilization_placeholder = st.empty()
    stabilization_placeholder.info("⚪ Waiting for Data")
with col2:
    sync_score = st.empty()
    sync_score.metric("Sync Accuracy", "0%")
with col3:
    kickoff_display = st.empty()
    kickoff_display.metric("Kickoff Found", "--:--")

# --- 5. UPDATED INPUT SECTION ---
st.divider()
input_mode = st.radio("Choose Match Report Source:", ["Paste Text", "Use URL"])

report_content = ""
if input_mode == "Paste Text":
    report_content = st.text_area("Paste Report Here", height=150)
else:
    report_url = st.text_input("Enter Match Report URL (e.g., ESPN, BBC Sport)")
    if report_url:
        with st.spinner("Scraping website..."):
            report_content = scrape_report_from_url(report_url)

video_file = st.file_uploader("Upload Match Video", type=['mp4'])

# --- 6. EXECUTION ---
if st.button("🚀 Start AI Sync & Cut"):
    if report_content and video_file:
        stabilization_placeholder.warning("🟡 Phase 1: Stabilizing Text...")
        events = parse_report(report_content)
        
        if not events:
            st.error("No timestamps found in the report content.")
        else:
            # (Rest of the kickoff detection and cutting logic remains the same)
            st.success(f"Found {len(events)} events! Proceeding to video scan...")
            # ... [Insert the rest of the previous code's logic here] ...
    else:
        st.error("Please provide both a report (text/URL) and a video.")
