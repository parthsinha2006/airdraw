# Real-Time ASL Static Sign Recognition

## Project Overview

This project implements a **real-time hand sign recognition system** inspired by **American Sign Language (ASL)** using **computer vision and machine learning**. The system detects a user's hand through a webcam, extracts landmark features using **MediaPipe**, and classifies the hand pose into predefined sign categories using a **K-Nearest Neighbors (KNN)** machine learning model.

The goal of the project is to demonstrate how **computer vision + machine learning** can be used to interpret human gestures in real time.

---

# Features

* Real-time webcam hand tracking
* Hand landmark detection using MediaPipe
* Custom dataset collection
* Feature extraction and normalization
* Machine learning classification using KNN
* Live sign prediction and visualization
* Modular project structure for easy extension

---

# Signs Recognized

The system currently recognizes **10 static hand signs**:

* YES
* NO
* OK
* STOP
* ILY (I Love You)
* THANKYOU
* FIST
* OPEN
* POINT
* PEACE

Note: These are **static hand poses inspired by ASL** used for gesture classification.

---

# Project Structure

```
sign_lang/
│
├── collect_data.py        # Collect training data from webcam
├── train_model.py         # Train ML classifier using collected data
├── predict_live.py        # Real-time sign recognition
├── features.py            # Feature extraction logic
│
├── data/                  # Stored training data (.npy files)
│   ├── YES.npy
│   ├── NO.npy
│   └── ...
│
├── model.pkl              # Saved trained ML model
├── hand_landmarker.task   # MediaPipe hand detection model
│
└── README.md
```

---

# Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Scikit-learn

---

# Installation

### 1. Clone or download the project

```
git clone <repository_url>
cd sign_lang
```

### 2. Create a virtual environment

```
python -m venv venv
```

Activate it:

Windows:

```
venv\Scripts\activate
```

---

### 3. Install dependencies

```
python -m pip install opencv-python mediapipe numpy scikit-learn
```

---

### 4. Download MediaPipe model

Download the hand landmark model:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Place it inside the project folder.

---

# How to Run the Project

## Step 1: Collect Dataset

Edit the label in `collect_data.py`:

```
SIGN_NAME = "YES"
```

Run:

```
python collect_data.py
```

Collect around **80 samples per sign**.

Repeat this process for all signs.

---

## Step 2: Train the Model

Once all data is collected:

```
python train_model.py
```

This will create:

```
model.pkl
```

---

## Step 3: Run Real-Time Prediction

Start the sign recognition system:

```
python predict_live.py
```

A webcam window will open and display the predicted sign.

Press **ESC** to exit.

---

# Feature Extraction Method

The system uses **21 hand landmarks detected by MediaPipe**.

Each landmark contains:

* x coordinate
* y coordinate

To improve robustness:

1. The **wrist landmark is used as the origin**
2. All landmarks are converted to **relative coordinates**
3. Coordinates are **normalized by hand size**

This results in a **42-dimensional feature vector** representing the hand pose.

---

# Machine Learning Model

The classifier used is **K-Nearest Neighbors (KNN)**.

Why KNN:

* Simple and interpretable
* Works well for geometric feature spaces
* No heavy training required

The model learns to classify hand poses based on similarity between feature vectors.

---

# Limitations

* Only supports **single-hand static signs**
* Works best with consistent lighting
* Dataset currently contains **single-user samples**
* Does not model **dynamic gestures or motion**

---

# Future Improvements

Possible extensions include:

* Support for **two-hand signs**
* Dynamic gesture recognition using **LSTM models**
* Larger multi-user datasets
* Prediction confidence scoring
* Temporal smoothing for stable predictions
* GUI interface for better user interaction

---

# Learning Outcomes

This project demonstrates concepts from:

* Computer Vision
* Feature Engineering
* Supervised Machine Learning
* Real-Time AI Systems
* Human–Computer Interaction

---

# Author

Parth Sinha
