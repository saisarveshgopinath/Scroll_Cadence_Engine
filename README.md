# 🚀 Scroll-Cadence Engine: AI Behavioral Telemetry & Attention Classifier

An executive-grade, real-time machine learning system that analyzes human biological interaction dynamics (trackpad/mouse scroll telemetry) to differentiate between active readers and automated skimmers or bots without relying on privacy-invasive webcams or static page timers.

---

## 🌟 Key Features

* **5-Second Dynamic Rolling Buffer:** Continuous, high-precision event collection capturing spatial Y-positions and millisecond timestamps using browser-native event listeners.
* **Directional & Statistical Feature Extraction:** Real-time processing via NumPy and Pandas to extract Inter-Scroll Intervals, Velocity ($px/ms$), Pause Variance (Standard Deviation), and Back-Scroll / Re-Reading Ratios.
* **Ensemble Machine Learning Pipeline:** Powered by a Random Forest Classifier trained on human scroll dynamics, achieving ~94% classification accuracy.
* **Humanized Adaptive UI:** Automatically triggers dark Focus Mode during high-engagement reading cycles and presents an automated executive summary box for rapid skimmers.
* **Real-Time Dual-Axis Chart.js Stream:** Visualizes live speed and pause variance dynamics over time.
* **Section-Level Attention Heatmaps:** Utilizes the browser `IntersectionObserver` API to track section dwell time and display dynamic engagement badges (`🔥 High Focus`, `⚡ Skimmed`, or `💤 Unread`).

---

## 🛠️ Tech Stack

* **Backend:** Python 3.10+, Flask, Scikit-Learn, Pandas, NumPy, Joblib
* **Frontend:** JavaScript (ES6+), HTML5, CSS3 (Glassmorphism & Custom Typography), Chart.js
* **Architecture:** RESTful Telemetry API, IntersectionObserver API

---

## 📁 Repository Structure

```text
Scroll_Cadence_Engine/
├── app.py              # Flask REST API for real-time inference
├── train_model.py      # Random Forest model training & feature extraction pipeline
├── model.pkl           # Trained Random Forest model binary
├── scroll_data.csv     # Raw behavioral telemetry dataset
├── index.html          # Humanized editorial dashboard & tracking script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
