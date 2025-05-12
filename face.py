import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io

# Page configuration
st.set_page_config(page_title="Face Detector", page_icon="🧠", layout="centered")

st.title("🧠 Face Detection App")
st.markdown("### Detect faces in images using the Viola-Jones algorithm 👇")

# 📝 Instructions
with st.expander("📌 How to use this app"):
    st.markdown("""
    1. **Upload an image containing one or more faces.**
    2. **Choose a color for the rectangle to draw around detected faces.**
    3. **Adjust detection accuracy via `scaleFactor` and `minNeighbors`.**
    4. **Click the 'Detect Faces' button to see results.**
    5. **Download the processed image if needed.**
    """)

# 🖼️ Upload Image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    st.image(image, caption="📷 Original Image", use_column_width=True)

    # 🎨 Rectangle Color
    rectangle_color = st.color_picker("🎨 Choose rectangle color", "#00FF00")
    b, g, r = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # 🔧 Detection Settings
    st.markdown("### ⚙️ Detection Settings:")
    scale_factor = st.slider("🔍 Scale Factor", min_value=1.05, max_value=2.5, value=1.1, step=0.05)
    min_neighbors = st.slider("👥 Min Neighbors", min_value=1, max_value=10, value=5)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 🚀 Detect Faces
    if st.button("🚀 Detect Faces"):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (b, g, r), 2)

        st.image(img_array, caption=f"🔎 {len(faces)} face(s) detected", use_column_width=True)

        # 💾 Save and Download Image
        result_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("💾 Download Image", data=byte_im, file_name="detected_faces.png", mime="image/png")
