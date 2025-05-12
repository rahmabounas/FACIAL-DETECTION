import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io

# Page configuration
st.set_page_config(page_title="Face Detector", page_icon="🧠", layout="centered")

st.title("🧠 Face Detection App")
st.markdown("Detect faces using the Viola-Jones algorithm")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Detection"])

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
        st.image(image, caption="📷 Original Image", use_column_width=True)

        rectangle_color = st.color_picker("🎨 Choose rectangle color", "#00FF00")
        b, g, r = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        scale_factor = st.slider("🔍 Scale Factor", min_value=1.05, max_value=2.5, value=1.1, step=0.05)
        min_neighbors = st.slider("👥 Min Neighbors", min_value=1, max_value=10, value=5)

        if st.button("🚀 Detect Faces"):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (b, g, r), 2)

            st.image(img_array, caption=f"🔎 {len(faces)} face(s) detected", use_column_width=True)

            result_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button("💾 Download Image", data=byte_im, file_name="detected_faces.png", mime="image/png")

with tab2:
    st.markdown("### Detect faces using your webcam")

    cam_option = st.selectbox("📷 Choose Camera", options=["Front Camera (0)", "Back Camera (1)"])
    camera_index = 1 if "Back" in cam_option else 0

    rectangle_color_web = st.color_picker("🎨 Rectangle Color (Webcam)", "#00FF00")
    b_web, g_web, r_web = tuple(int(rectangle_color_web.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    scale_factor_web = st.slider("🔍 Scale Factor (Webcam)", min_value=1.05, max_value=2.5, value=1.1, step=0.05)
    min_neighbors_web = st.slider("👥 Min Neighbors (Webcam)", min_value=1, max_value=10, value=5)

    start_cam = st.button("🎥 Start Camera")

    if start_cam:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.warning("❗ Couldn't access selected camera. Falling back to front camera.")
            cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()
        capture_btn = st.button("📸 Capture Photo")

        captured = False
        captured_frame = None

        while cap.isOpened() and not captured:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor_web, minNeighbors=min_neighbors_web)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (b_web, g_web, r_web), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if capture_btn:
                captured_frame = frame_rgb.copy()
                captured = True
                break

        cap.release()

        if captured_frame is not None:
            st.image(captured_frame, caption="📸 Captured Photo with Detected Faces", use_column_width=True)
            result_img = Image.fromarray(captured_frame)
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("💾 Download Captured Image", data=byte_im, file_name="captured_faces.png", mime="image/png")