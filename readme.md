# 🧠 ASL Sign Language Detection Using Deep Learning

> **Internship Project – Unified Mentor Pvt Ltd**  
> *Intern: [Rudraksh Sachdeva]*  
> *Internship Duration: [10-06-2025] to [10-07-2025]*  
> *GitHub Repository: [https://github.com/Rudrakshsachdev/asl-sign-language-detection.git]*

---

## 📌 Overview

This project focuses on the **real-time detection of American Sign Language (ASL) alphabets** using a webcam and a Convolutional Neural Network (CNN). It aims to bridge communication gaps between the hearing-impaired and the general public. The system recognizes hand gestures corresponding to letters A-Z, plus 'space', 'delete', and 'nothing', and converts them into readable text on screen — with optional **text-to-speech functionality**.

---

## 🎯 Objective

- Build a machine learning model that can classify ASL hand gestures in real-time.
- Design a system to predict characters and form readable sentences.
- Enhance accessibility using TTS (Text-To-Speech) for audio output.

---

## 🧠 Key Features

- 📷 Real-time webcam integration using OpenCV  
- 🧠 CNN-based classifier trained on a labeled ASL dataset  
- 🗣️ Converts predicted characters into spoken words using pyttsx3  
- 📄 Modular, maintainable code with clear documentation  
- 📊 Model achieves high accuracy (~99.7% on validation data)

---

## 🗂️ Project Structure

ASL_Sign_Detection_Project/
├── data/ # ASL dataset (excluded in GitHub)
│ └── rawData/
│ └── asl_alphabet_train/
├── src/ # Source scripts
│ ├── Preprocessing.py # Data loading and preprocessing
│ ├── TrainModel.py # CNN model building and training
│ ├── EvaluateModel.py # Model performance evaluation
│ ├── RealTimeASL.py # Real-time ASL detection and sentence building
│ └── models/
│ └── asl_model.h5 # Saved trained model
├── .gitignore
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # You are here


---

## 🧪 Dataset

- **Dataset**: [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Format**: Image folders per alphabet (e.g., `A/`, `B/`, ..., `space/`)
- **Size**: ~87,000 images, 200x200 resolution
- 📌 *Place the dataset inside:*


---

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.10** | Programming language |
| **TensorFlow / Keras** | Deep learning model |
| **OpenCV** | Image capture & processing |
| **NumPy** | Numerical operations |
| **scikit-learn** | Data splitting |
| **pyttsx3** | Text-to-speech synthesis |

---

## ⚙️ Setup Instructions

1. Clone the Repository
git clone https://github.com/your-username/asl-sign-language-detection.git
cd asl-sign-language-detection

2. Create Virtual Environment (Optional)
python -m venv .venv
.venv\Scripts\activate (for Windows)

3. Install Dependencies
pip install -r requirements.txt

4. Prepare the Dataset
Download dataset from Kaggle and extract it to:
data/rawData/asl_alphabet_train/

---

## Run the Project
Step 1: Preprocess Dataset
python src/Preprocessing.py

Step 2: Train the Model
python src/TrainModel.py

Step 3: Evaluate the Model
python src/EvaluateModel.py

Step 4: Run Real-Time Detection with TTS
python src/DetectSign.py

---

## Model Architecture
The CNN model contains the following layers:

1. 3× Conv2D + MaxPooling2D

2. Flatten

3. Dense layers with ReLU activations

4. Final output layer with softmax for 29-class classification

---

## Performance
1. Validation Accuracy: ~99.71%
2. Loss: ~0.0122
3. Real-time FPS: ~10–15 FPS (system-dependent)

---

## Sentence & Text-to-Speech
1. System constructs a sentence from recognized characters.

2. If a gesture is held for a few seconds, it is added to the sentence.

3. Press Enter to trigger pyttsx3 to speak the constructed sentence.

---

## Advanced Features (Implemented)
✅ Real-time sentence formation

✅ Text-to-speech support

✅ Timer-based gesture detection

---

## Future Enhancements
1. Deploy as a Flask or Streamlit web app

2. Integrate mobile camera APIs

3. Use MobileNet or EfficientNet for better accuracy

4. Include ASL word recognition

