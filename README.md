
# Deep Crack Detection 🚀

A deep learning project that detects cracks in structures using MobileNet architecture and transfer learning.

## 📂 About

Deep Crack Detection is an image processing project that uses **MobileNet** architecture and **transfer learning** to identify and detect cracks in structural images. The model is trained using labeled crack images and can be used for real-time crack detection in various environments.

---

## 🛠️ Tech Stack

- **Python**: Programming language
- **TensorFlow & Keras**: Deep learning libraries for model training
- **OpenCV**: Image processing
- **Django**: Backend framework for running the web app
- **Git LFS**: For handling large model files

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Deep_Crack.git
cd Deep_Crack
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Setup Git LFS (for large model files):

```bash
git lfs install
git lfs pull
```

---

## 🚀 How to Run

1. Run the Django server:

```bash
python manage.py runserver
```

2. Open a browser and go to `http://127.0.0.1:8000/` to start using the crack detection model.

---

## 🧠 Features

- **Model Training**: Train the crack detection model using MobileNet and transfer learning.
- **Image Prediction**: Upload images to the web interface for crack detection.
- **Real-time Crack Visualization**: Visualize cracks detected in images with bounding boxes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- **TensorFlow** and **Keras** for deep learning model development.
- **OpenCV** for image processing tasks.
- **Django** for the web framework.
- **GitHub** for hosting the project.

---

### 📌 Notes
- Make sure you have **Git LFS** installed to access large `.h5` model files.
- If models are missing, run `git lfs pull`.
