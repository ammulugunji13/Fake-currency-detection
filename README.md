# 💵 Fake Currency Detection - AI Project | 98.7% Accuracy

> An Intelligent Deep Learning System to Detect Fake Indian Currency Notes using Computer Vision.

Developed by **Anusha Gunji** | B.Tech AI/ML Enthusiast | Andhra Pradesh, India

### 📌 About Project
This project is my Final Year Major Project. It helps common people and shopkeepers to identify fake currency notes instantly by just uploading an image.

### 🧠 How it Works
1. **Upload:** User uploads currency note image
2. **Preprocess:** System converts to Grayscale, detects edges & watermarks using OpenCV
3. **Predict:** Trained CNN Model (TensorFlow/Keras) analyzes the note
4. **Result:** Shows Real or Fake with Confidence Score

### 💻 Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow, Keras, CNN
- **Computer Vision:** OpenCV, NumPy
- **Frontend:** Streamlit
- **Dataset:** 10,000+ Real & Fake Indian Notes (RBI Dataset)

### 📊 Model Performance
- **Accuracy:** 98.7%
- **Precision:** 98.2%
- **Model Used:** Custom CNN with 5 Convolutional Layers

### 📁 Files in this Repo
- `app.py` - Main Streamlit Application
- `model.py` - CNN Model Architecture
- `train.py` - Model Training Code
- `predict.py` - Prediction Logic
- `requirements.txt` - All Dependencies

### 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py

### 🔮 Future Scope
- Mobile App for real-time camera detection
- Support for 500, 200, 100 denominations
- Multi-currency support
