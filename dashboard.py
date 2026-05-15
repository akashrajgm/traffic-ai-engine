import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. Page Configuration & Dark Theme CSS
st.set_page_config(page_title="Traffic Vision AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; width: 100%; }
    .stAlert { background-color: #1e1e1e; border: 1px solid #2e7d32; color: #00ff00; }
    h1 { color: #00ff00; font-family: 'Courier New', Courier, monospace; text-align: center; text-shadow: 2px 2px #000; }
    .stImage { border-radius: 15px; border: 2px solid #333; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 TRAFFIC VISION AI ENGINE")
st.markdown("<h3 style='text-align: center; color: #888;'>Precision Analytics • Real-Time Detection</h3>", unsafe_allow_html=True)

# Tabs for Image and Video
tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Analytics"])

@st.cache_resource
def load_model():
    return YOLO('yolov8m.pt')

model = load_model()

# --- TAB 1: IMAGE DETECTION ---
with tab1:
    img_file = st.file_uploader("Drop a photo here...", type=['jpg', 'jpeg', 'png'], key="img")
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Source", use_container_width=True)
        
        if st.button("RUN AI DETECTION", key="run_img"):
            with st.spinner("⚡ Processing pixels..."):
                results = model(image)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="AI Output", use_container_width=True)
                st.balloons()

# --- TAB 2: VIDEO ANALYTICS ---
with tab2:
    vid_file = st.file_uploader("Drop a video here...", type=['mp4', 'mov', 'avi'], key="vid")
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(vid_file.read())
        
        if st.button("START ENGINE", key="run_vid"):
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            output_path = "processed_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            count = 0
            ids = set()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                line_y = int(height * 0.65)
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.45, verbose=False)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    for box, tid in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        if y1 < line_y < y2 and tid not in ids:
                            ids.add(tid)
                            count += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
                cv2.putText(frame, f"AI COUNT: {count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                out.write(frame)
                
                current_frame += 1
                progress_bar.progress(current_frame / total_frames)
                status_text.text(f"Analyzing Frame: {current_frame}/{total_frames}")

            cap.release()
            out.release()
            st.success(f"ANALYSIS COMPLETE: {count} Vehicles Detected")
            st.video(output_path)
            st.download_button("Download Report", open(output_path, 'rb'), file_name="Traffic_Analysis.mp4")
