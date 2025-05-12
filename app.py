import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io
import time
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.state import get_session_state
def rerun():
    raise RerunException(get_session_state())

# Page configuration
st.set_page_config(page_title="Face Detector", page_icon="🧠", layout="centered")

st.title("🧠 Face Detection App")
st.markdown("Detect faces using the Viola-Jones algorithm")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Detection"])

# ---------------------- Tab 1: Upload Image ----------------------
with tab1:
    st.markdown("### Detect faces in an uploaded image")

    # 📝 Instructions
    with st.expander("📌 How to use this tab"):
        st.markdown("""
        1. **Upload an image containing one or more faces.**
        2. **Choose a color for the rectangle to draw around detected faces.**
        3. **Adjust detection accuracy via `scaleFactor` and `minNeighbors`.**
        4. **Click the 'Detect Faces' button to see results.**
        5. **Download the processed image if needed.**
        """)

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        st.image(image, caption="📷 Original Image", use_container_width=True)

        rectangle_color = st.color_picker("🎨 Choose rectangle color", "#00FF00")
        b, g, r = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        scale_factor = st.slider("🔍 Scale Factor", min_value=1.05, max_value=2.5, value=1.1, step=0.05)
        min_neighbors = st.slider("👥 Min Neighbors", min_value=1, max_value=10, value=5)

        if st.button("🚀 Detect Faces"):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (b, g, r), 2)

            st.image(img_array, caption=f"🔎 {len(faces)} face(s) detected", use_container_width=True)

            result_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button("💾 Download Image", data=byte_im, file_name="detected_faces.png", mime="image/png")

# ---------------------- Tab 2: Webcam Detection ----------------------
with tab2:
    st.markdown("### Detect faces using your webcam")

    cam_option = st.selectbox("📷 Choose Camera", options=["Back Camera (1)", "Front Camera (0)"])
    camera_index = 1 if "Front" in cam_option else 0

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "captured_frame" not in st.session_state:
        st.session_state.captured_frame = None

    start = st.button("🎥 Start Camera")
    stop = st.button("🛑 Stop Camera")

    if start:
        st.session_state.camera_active = True
        st.session_state.captured_frame = None
    if stop:
        st.session_state.camera_active = False

    frame_placeholder = st.empty()
    capture_placeholder = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.warning("❗ Couldn't access selected camera. Falling back to front camera.")
            cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Save frame for capture
            st.session_state.last_frame = frame_rgb
        cap.release()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 Capture Photo"):
                st.session_state.captured_frame = st.session_state.last_frame

        with col2:
            if st.button("🔄 Refresh"):
                time.sleep(0.1)
                rerun()

    if st.session_state.captured_frame is not None:
        capture_placeholder.image(st.session_state.captured_frame, caption="📸 Captured Photo", use_container_width=True)
        result_img = Image.fromarray(st.session_state.captured_frame)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("💾 Download Captured Image", data=byte_im, file_name="captured_faces.png", mime="image/png")
