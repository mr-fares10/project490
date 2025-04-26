# Project490: Fighter Jet Video Analysis and Tracking

This repository provides a complete system for detecting, tracking, and predicting the 3D positions of fighter jets from both images and videos. It combines deep learning models with depth estimation, allowing for real-time analysis and future trajectory prediction. The system includes both a web interface and Google Colab-compatible scripts.

---

## Components Overview

### 1. CenterXYZDetection
**Single-Image Analysis Engine**

This script analyzes still images of fighter jets using:
- **Mask R-CNN** (ResNet50 backbone) for object segmentation.
- **MiDaS** for monocular depth estimation and 3D positioning.

**Features:**
- Automatic CUDA/CPU selection.
- Customizable camera and jet parameters.
- 3D position extraction based on image depth.
- Camera intrinsic parameter calculation.
- Visual and numerical outputs of jet position.

---

### 2. VideoXYZ1FramePerSecond
**Low-Frequency Video Analysis**

- Extracts video frames at 1 FPS (configurable).
- Applies the same detection pipeline as `CenterXYZDetection`.
- Tracks 3D positions (x, y, z) over time.
- Outputs:
  - Annotated videos.
  - Trajectory CSV files.
  - 2D and 3D motion visualizations.

---

### 3. VideoXYZHighFrameRateProcessing60fps
**High-Frequency Video Processing**

- Processes videos at high frame rates (up to 60 FPS or native).
- Outputs three synchronized videos:
  - Annotated tracking video.
  - Segmentation mask video.
  - Depth map video.
- Maintains continuous jet tracking.
- Optimized for efficient resource use.

---

### 4. Fighter Jet LSTM Model Training Pipeline
- Extracts jet trajectory data from multiple videos.
- Trains LSTM models for future 3D position prediction.
- Uses object detection, depth estimation, and calibration.
- Designed for Google Colab, with Google Drive integration for file handling.

---

### 5. Fighter Jet Position Prediction Inference Pipeline
- Uses a pre-trained LSTM to predict future positions of fighter jets from video.
- Performs detection, depth estimation, and trajectory prediction.
- Outputs:
  - Annotated videos with predictions.
  - CSVs with 3D trajectories.
  - Visual comparisons between actual and predicted paths.
  - Performance summaries.
- **Note:** This script can be run directly in **Google Colab**, allowing users to upload any video and obtain the same outputs as the web interface.

---

# Web Interface for Video Analysis

A Flask-based web interface allows users to upload videos and receive 3D position predictions.

### Key Components:
- **mainweb.py**: Launches the Flask web service.
- **advanced_processor.py**: Backend processor called by `mainweb.py` for video analysis.
- **web_app/**:
  - Contains all web interface files.
  - **enhanced.html**: Frontend for video upload and result viewing.
- **examples/**: Graphics and assets used for frontend design.
- **Dockerfile**: For Docker-based deployment of the entire web service.

---

## Features

- Web-based video upload and real-time fighter jet tracking.
- Supports various video formats: MP4, AVI, MOV.
- Real-time progress tracking during processing.
- Drag-and-drop functionality for easy file upload.
- Output includes:
  - Annotated video with 3D position overlays.
  - CSV table of 3D positions with timestamps.
  - Visual performance summaries.

---

## Using Google Colab

You can run **Fighter Jet Position Prediction Inference Pipeline.py** in **Google Colab** to:
- Upload any video.
- Run detection, depth estimation, and LSTM-based prediction.
- Obtain:
  - Annotated videos.
  - 3D trajectory CSV files.
  - Visual comparison plots.
- This allows for easy cloud-based processing without using the web interface.

---

## Prerequisites

- Python 3.8 or higher
- Poetry (for dependency management)
- Docker (optional, for deployment)
- CUDA-compatible GPU (optional, for faster local processing)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mr-fares10/project490.git
cd project490
