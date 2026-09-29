# 🚗 Real-Time Driver Drowsiness Detection System
### Comparative Performance Analysis: Custom Deep CNN vs. Pretrained MobileNetV2 (Transfer Learning)

---
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Dataset: MRL Eye Dataset](https://img.shields.io/badge/Dataset-MRL%20Eye%20(85k%20Images)-blueviolet.svg)](http://mrl.cs.vsb.cz/eyedataset)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📸 Live Application Demo

<!-- ========================================================================= -->
<!-- PLACEHOLDER FOR DEMO SCREENSHOT:                                          -->
<!-- Save your webcam demo screenshot as 'demo_screenshot.png' in this folder  -->
<!-- ========================================================================= -->
<div align="center">
  <img src="demo_screenshot.png" alt="Driver Drowsiness Detection System Live Demo" width="800" onerror="this.src='https://raw.githubusercontent.com/tanvi-09112005/Driver_Drowsiness_Detection/main/haar_cascade_files/demo_placeholder.png'; this.onerror=null;" />
  <p><i>Figure 1: Real-time driver face tracking, ocular state classification (Open vs. Closed), dynamic biometric eye anchoring, and auditory drowsiness alert HUD.</i></p>
</div>

---

## 📌 Project Overview
Driver fatigue and microsleep episodes are among the leading causes of fatal roadway collisions worldwide. This project delivers an end-to-end, computer-vision-based **Driver Drowsiness Detection System** designed for edge deployment and in-cabin automotive safety (ADAS).

The system monitors live camera streams, continuously tracks the driver's face and eyes using dynamic biometric anchors, classifies eye states (Open vs. Closed) via Deep Learning, tracks the temporal eye-closure duration (**PERCLOS** metric), and sounds an emergency auditory alarm when eyes remain closed past a critical safety threshold ($1.2\text{ s}$).

### 🔬 Core Academic Task: Model Comparison
As part of the **DS Lab MiniProject Assignment**:
1. **Model 1 (Designed from Scratch)**: A bespoke **Custom Deep Convolutional Neural Network (CNN)** optimized for ocular spatial feature extraction (eyelid boundaries, pupil occlusion).
2. **Model 2 (Pretrained Architecture)**: A fine-tuned **MobileNetV2** transfer learning model with an ImageNet backbone, utilizing depthwise separable convolutions and inverted residual bottlenecks.
3. **Rigorous Benchmark**: Side-by-side comparative analysis evaluating Accuracy, Precision, Recall/Sensitivity (the safety-critical metric), Specificity, F1-Score, ROC-AUC, model footprint, and live inference latency (FPS/ms).

---

## 📂 Repository Structure
```text
Driver_Drowsiness_Detection/
├── Driver_Drowsiness_Detection.ipynb       # Local Jupyter Notebook (Fast stratified training)
├── Driver_Drowsiness_Detection_Colab_Full.ipynb # Google Colab GPU Edition (Full 85k dataset)
├── realtime_detection.py                  # v3.0 Live webcam detection application
├── generate_sample_dataset.py             # Offline test dataset generator
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git ignore rules (excludes large datasets)
├── README.md                              # Project documentation & PPT guide
├── demo_screenshot.png                    # [Placeholder] Put your demo screenshot here!
├── alarm.wav                              # Continuous emergency auditory alert sound
├── haar_cascade_files/                    # Pre-trained Haar Cascade XML classifiers
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
└── models/                                # Trained model checkpoints
    ├── driver_drowsiness_custom_full85k.keras   (1.4 MB - Custom Deep CNN)
    ├── driver_drowsiness_mobilenet_full85k.keras (11.6 MB - MobileNetV2)
    └── model.h5                                 (17.3 MB - Reference baseline)
```

---

## 📊 Dataset Description
* **Dataset Name:** [MRL Eye Dataset](http://mrl.cs.vsb.cz/eyedataset)
* **Kaggle Mirror:** [Drowsiness Dataset (MRL Eyes)](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)
* **Scale:** ~84,898 grayscale and infrared images captured across 37 unique human subjects under diverse lighting conditions, head poses, eyeglasses, and sensor reflections.

### Filename Metadata Syntax:
Each file encodes 8 categorical attributes separated by underscores (`_`):
$$\text{SubjectID\_ImageID\_Gender\_Glasses\_EyeState\_Reflections\_Lighting\_SensorID.png}$$

* **Target Label (Index 4):**
  * `0`: **Closed Eye** (Drowsy State)
  * `1`: **Open Eye** (Alert State)
* **Safety Context:** In automotive safety, a **False Negative (classifying a closed eye as open)** is life-threatening. Therefore, **Recall for Closed Eyes** is our primary optimization metric.

---

## 🧠 Model Architectures & Algorithms

```
                            INPUT IMAGE (64 x 64 x 3)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│   Model 1: Custom Deep CNN    │     │      Model 2: MobileNetV2             │
│    (Designed from Scratch)    │     │       (Transfer Learning)             │
├───────────────────────────────┤     ├───────────────────────────────────────┤
│ • 3 Conv Blocks (32, 64, 128) │     │ • Pretrained ImageNet Backbone        │
│ • Batch Normalization + ReLU  │     │ • Depthwise Separable Convolutions    │
│ • Spatial Dropout (0.25)      │     │ • Inverted Residual Bottlenecks       │
│ • GlobalAveragePooling2D      │     │ • Frozen Feature Extraction Base      │
│ • Dense(128, ReLU) + Dropout  │     │ • GlobalAvgPool + Dense(128) + Dropout│
│ • Dense(2, Softmax)           │     │ • Dense(2, Softmax)                   │
├───────────────────────────────┤     ├───────────────────────────────────────┤
│ ~110,914 Parameters           │     │ ~2,422,722 Parameters                 │
│ Disk Size: ~1.4 MB            │     │ Disk Size: ~11.6 MB                   │
│ Latency: ~14 ms (~70 FPS)     │     │ Latency: ~31 ms (~32 FPS)             │
│ Optimized for Edge/Embedded   │     │ High Generalization under Var. Light  │
└───────────────────────────────┘     └───────────────────────────────────────┘
```

---

## 📈 Comparative Results Summary

| Evaluation Metric | Model 1: Custom Deep CNN | Model 2: Pretrained MobileNetV2 | Significance in Automotive Safety |
|:---|:---:|:---:|:---|
| **Overall Accuracy** | **97.8%** | **98.4%** | Both achieve high baseline discrimination |
| **Precision** | **97.4%** | **98.1%** | Minimizes nuisance false alarms |
| **Recall (Closed Eyes) [CRITICAL]** | **98.1%** | **98.8%** | **Safety-Critical:** Catches falling-asleep episodes |
| **Specificity (Open Eyes)** | **97.5%** | **98.0%** | Accurately identifies alert driving |
| **F1-Score** | **0.977** | **0.984** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.993** | **0.998** | High separation across all decision thresholds |
| **Model Disk Size** | **~1.4 MB** | **~11.6 MB** | **Custom CNN is $\approx 8\times$ smaller** (microcontroller flash) |
| **Inference Latency (CPU)** | **~14.2 ms / frame** | **~31.5 ms / frame** | **Custom CNN is $\approx 2.2\times$ faster** |
| **Throughput (FPS)** | **~70 FPS** | **~32 FPS** | Custom CNN easily exceeds 30 FPS video standard |

---

## 🚀 How to Run

### 1. Real-Time Live Webcam Application
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/tanvi-09112005/Driver_Drowsiness_Detection.git
   cd Driver_Drowsiness_Detection
   pip install -r requirements.txt
   ```
2. Launch the application:
   ```bash
   python realtime_detection.py
   ```
   *(By default, uses the trained 85k Custom CNN: `models/driver_drowsiness_custom_full85k.keras`)*

3. To run with MobileNetV2:
   ```bash
   python realtime_detection.py --model models/driver_drowsiness_mobilenet_full85k.keras
   ```

4. **Features & Controls:**
   * **Face & Tilt Tracking:** Head-tracking memory retains lock even during head tilts.
   * **PIP Preview:** Real-time eye crop inspection in the bottom-right corner.
   * **Audio Alarm:** Plays continuous emergency alarm when eyes remain shut $\ge 1.2\text{ s}$.
   * **Exit:** Press **`q`** or **`ESC`** to terminate.

---

### 2. Google Colab GPU Notebook (Full 85k Dataset)
Open [`Driver_Drowsiness_Detection_Colab_Full.ipynb`](Driver_Drowsiness_Detection_Colab_Full.ipynb) in **Google Colab**, enable GPU (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**), and click **Run all**. It downloads the dataset, trains both models with GPU streaming (`tf.data.AUTOTUNE`), and exports the final weights.

---

## 📽️ PPT Presentation Outline (DS Lab Submission)

When submitting your **4 to 5 slide presentation**:
* **Slide 1: Problem Statement**
  * Real-world statistics: 100,000+ crashes annually caused by driver drowsiness (NHTSA/WHO).
  * Objective: Autonomous, non-intrusive vision system detecting prolonged eye closure in real-time to alert drivers before accidents happen.
* **Slide 2: Dataset Description & Analysis**
  * Dataset: MRL Eye Dataset (84,898 images across 37 subjects).
  * Link: [http://mrl.cs.vsb.cz/eyedataset](http://mrl.cs.vsb.cz/eyedataset).
  * Variations: Infrared/visible light, eyeglasses, reflections, and varied illumination.
* **Slide 3: Model 1 — Custom Deep CNN (Designed from Scratch)**
  * 3 Conv2D blocks (32, 64, 128) + Batch Normalization + ReLU + Spatial Dropout.
  * GlobalAveragePooling2D + Dense(128) + Softmax output (~110K params, 1.4 MB).
* **Slide 4: Model 2 — Transfer Learning with MobileNetV2**
  * Pretrained on ImageNet-1k with depthwise separable convolutions.
  * Inverted residual bottlenecks for robust feature extraction under complex lighting.
* **Slide 5: Results & Comparative Analysis**
  * Comparative performance table (Accuracy, Recall, Size, Latency, FPS).
  * Trade-off summary: MobileNetV2 yields top accuracy (98.4%), while Custom CNN offers superior edge speed (70 FPS vs 32 FPS) and compact size (1.4 MB).
