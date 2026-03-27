# Fall-detection-LSTM-YOLOv11-pose
An end-to-end deep learning system that detects human falls from video using pose estimation and temporal sequence modeling.
👉 Includes a live demo deployed on Hugging Face.
👉 Designed for healthcare monitoring, elderly safety, and smart surveillance.

## Problem

Falls are a major health risk, especially for elderly people.
According to global health reports, **~37 million falls each year are severe enough to require medical attention**.
Early detection can significantly reduce injury severity and response time.

## 💡 Solution

This project combines spatial and temporal deep learning:

- YOLO pose estimation → extracts 17 human keypoints per frame
- LSTM neural network → models motion over time
- Sliding window inference → detects fall events in sequences

The system outputs:
- Fall probability over time
- Final classification: FALL / NO FALL

## ⚙️ System Pipeline

1. Input video
2. Extract human pose (17 keypoints) using YOLO11s-pose
3. Convert keypoints into time-series data (x, y, confidence)
4. Feed sequences into a 3-layer LSTM model
5. Compute fall probability for each time window
6. Apply decision logic for final classification

## 🎥 Demo

🔗 Live Demo: [YOUR HUGGING FACE LINK]

The demo allows users to:
- Upload a video
- Visualize fall probability timeline
- Get final prediction

Example outputs include:
- Probability vs time plot
- Fall detection summary

## 📊 Results

The model achieves strong performance despite a small and imbalanced dataset:

- Accuracy: **91.89%**
- Fall Precision: **84.09%**
- Fall Recall: **88.10%**
- Fall F1-score: **86.05%**

Confusion Matrix:

|               | Pred: Non-Fall | Pred: Fall |
|---------------|----------------|------------|
| True Non-Fall | 99             | 7          |
| True Fall     | 5              | 37         |

The model maintains a good balance between detecting falls and minimizing false alarms.

## Dataset

- Source: GMDCSA24 (public dataset)
- Characteristics:
  - Small dataset
  - Class imbalance (fewer fall samples)


