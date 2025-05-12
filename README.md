# 🧠 Face Detection App (Streamlit + Viola-Jones)

A simple, interactive face detection web app built with **Streamlit** and **OpenCV**, using the classic **Viola-Jones algorithm**. Detect faces in images, customize detection settings, choose bounding box colors, and download results easily.

---

## 🚀 Features

- 📤 Upload images (JPEG, PNG).
- 🎨 Select rectangle color for face highlights.
- ⚙️ Adjust detection parameters:
  - `scaleFactor` (detection precision)
  - `minNeighbors` (confidence level)
- 👁️ View processed images with faces detected.
- 💾 Download the result directly.

---

## 🛠️ Tech Stack

- [Python 3.7+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Pillow (PIL)](https://pillow.readthedocs.io/)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-detection-app.git
   cd face-detection-app

2. **Run the app**
   ```bash
   streamlit run app.py

## 🖼️ Screenshot

![Screenshot](screenshot.png)

---

## 📌 How to Use

1. **Launch the app locally** with `streamlit run app.py`.
2. **Upload an image** containing one or more faces.
3. **Pick the rectangle color** using the color picker.
4. **Adjust detection settings** using sliders:
   - **`scaleFactor`**: controls scale step for detection.
   - **`minNeighbors`**: sets the strictness of detection.
5. **Click "Detect Faces"** to see the result.
6. **Click "Download Image"** to save the output.

---

## ✅ Example Use Cases

- **Quick face detection** for photos.
- **Exploring classical CV algorithms**.
- **Prototype for more advanced detection systems**.
- **Educational or demonstration purposes**.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🌐 Website

You can access the live web app here: [facedetection](https://facedetectionj.streamlit.app/)

---

## 🧑‍💻 Author

**Rahma** – Data Science Student  
📫 Reach out via issues or discussions for questions or suggestions.
