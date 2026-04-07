# 🎤 Speech Emotion Recognition using CNN (RAVDESS)

This project implements a **Speech Emotion Recognition (SER)** system using a **Convolutional Neural Network (CNN)** trained on the **RAVDESS dataset**. The model analyzes speech audio by converting it into log-Mel spectrograms and classifies emotions into multiple categories.

---

## 📌 Overview

Speech emotion recognition is a challenging task due to subtle acoustic differences between emotions and limited dataset size. This project focuses on building a **robust and balanced CNN model** that avoids common issues like class imbalance and model collapse.

The system uses:

* Log-Mel spectrograms for feature representation
* CNN for feature extraction and classification
* Speaker-independent evaluation for realistic performance

---

## 🎯 Objectives

* Detect emotions from speech audio signals
* Build a CNN model capable of handling small datasets
* Ensure balanced predictions across all emotion classes
* Avoid class collapse and improve minority class learning

---

## 🧠 Model Architecture (CNN)

* Input: Log-Mel Spectrogram (3-second audio)
* Convolution Blocks:

  * Conv2D + ReLU + BatchNorm
  * Pooling + Dropout
* Global Average Pooling
* Fully Connected Layer
* Output Layer: Softmax (8 classes)

---

## 🎭 Emotion Classes

The model predicts the following 8 emotions:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## 🔄 System Pipeline

1. Load RAVDESS dataset
2. Filter valid speech samples
3. Preprocess audio:

   * Convert to mono
   * Trim silence
   * Fix duration (3 sec)
4. Generate log-Mel spectrogram
5. Normalize features
6. Train CNN model
7. Predict emotion using softmax
8. Evaluate using accuracy, F1-score, confusion matrix

---

## 📊 Dataset

* Dataset: **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
* Total samples: ~1440
* Audio duration: ~3 seconds
* Sampling rate: 22050 Hz
* Type: Speech-only (songs excluded)

📎 Source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## 📈 Results

* **Accuracy:** 43.8%
* **Macro F1-score:** 0.386
* **Evaluation:** Speaker-independent

👉 The model achieves **balanced predictions across all classes**, avoiding bias toward dominant emotions.

---

## ⚙️ Tech Stack

* Python 3.10+
* TensorFlow / Keras
* Librosa
* NumPy, Pandas
* Scikit-learn
* Matplotlib, Seaborn
* Gradio (for UI)

---

## 💡 Key Improvements

Compared to earlier versions, the final model:

* Fixes class imbalance using **sample-wise weighting**
* Removes harmful augmentations (MixUp)
* Reduces over-regularization
* Improves model capacity (Dense layer tuning)
* Uses controlled SpecAugment

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/pranjalimishra2005/speech_emotion_recognition.git
cd speech-emotion-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run model

```bash
python speech_emotion_recognition_v4.py
```

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Confusion Matrix

---

## 🌐 Real-Time Interface

A **Gradio-based interface** is implemented for:

* Uploading audio files
* Recording via microphone
* Predicting emotion in real-time

---

## ⚠️ Limitations

* Small dataset (~1440 samples)
* Similar emotions are hard to distinguish (e.g., happy vs surprised)
* Acted dataset (not real-world speech)
* Limited generalization across datasets

---

## 🔮 Future Work

* Use larger datasets (CREMA-D, EmoDB)
* Apply pretrained models (wav2vec, HuBERT)
* Add attention mechanisms
* Cross-dataset evaluation
* Real-time deployment improvements

---

## 👩‍💻 Author

* Pranjali Mishra


---

## 📚 Reference

This README is based on the project report:


---

## ⭐ Note

This project demonstrates how **careful debugging and training strategy improvements** can significantly impact model performance, even more than increasing model complexity.
