# Project490: Fighter Jet Video Analysis and Tracking

This repository contains a comprehensive framework for automated fighter jet detection, tracking, and analysis from both still images and video footage.

## Components

### 1. CenterXYZDetection
**Single-Image Analysis Engine**

This core Python script performs sophisticated analysis of fighter jet photographs, leveraging:
- Mask R-CNN with ResNet50 backbone for object segmentation
- MiDaS for depth estimation and 3D positioning

Key features:
- Automatic device optimization (CUDA/CPU)
- Robust model loading with fallback mechanisms
- User-configurable camera and aircraft parameters
- Precise camera intrinsic parameter calculation
- Accurate jet identification and dimension measurement
- Multiple calibration techniques
- Detailed visualization and measurement outputs

### 2. VideoXYZ1FramePerSecond
**Low-Frequency Video Analysis**

Extends the image analysis capabilities to video with selective frame sampling:
- Extracts frames at 1 frame per second (configurable)
- Applies the full detection algorithm to each frame
- Tracks 3D position (x, y, z) across time
- Generates annotated output video with tracking data
- Records trajectory data in CSV format
- Creates 2D and 3D visualizations of aircraft path

### 3. VideoXYZHighFrameRateprocessing60fps**

Optimized for continuous tracking at high frame rates:
- Processes at up to 60fps or native video frame rate
- Generates three synchronized output videos:
  * Annotated tracking video with measurements
  * Dedicated segmentation mask video
  * Separate depth map video
- Maintains continuous object tracking
- Efficiently manages resources for high volume processing
- Provides comprehensive motion analysis
### 4.Fighter Jet LSTM Model Training Pipeline
This script provides a comprehensive training pipeline for predicting the 3D position of fighter jets using LSTM-based deep learning models. It supports trajectory extraction from multiple videos, preprocessing, model training, evaluation, and result export. The pipeline leverages object detection (Mask R-CNN), depth estimation (MiDaS), and real-world camera calibration to generate time-series data, which is then used to train an LSTM model to predict future jet positions. It is designed for use in Colab and integrates with Google Drive for seamless file handling and storage.
### 5.Fighter Jet Position Prediction Inference Pipeline.py
This script performs inference using a pre-trained LSTM model to predict the 3D trajectory of a fighter jet in video frames. It includes object detection (Mask R-CNN), monocular depth estimation (MiDaS), and camera calibration for converting 2D centroids to 3D coordinates. The model predicts future jet positions based on historical movement patterns. The script generates annotated videos, trajectory CSVs, visual comparisons between actual and predicted paths, and performance summaries. Designed for use in Google Colab, it supports both video and model uploads and automatically compresses results for easy download.

# Fighter Jet Position Prediction Project

This project provides a web interface for predicting 3D positions of fighter jets in video footage using deep learning models.

## Features

- Web-based interface for video upload and processing
- Real-time progress tracking during video processing
- Support for multiple video formats (MP4, AVI, MOV)
- Position predictions with timestamps
- Modern, responsive UI with drag-and-drop support

## Prerequisites

- Python 3.8 or higher
- Poetry for dependency management
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd project490
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Running the Application

1. Start the Flask development server:
```bash
cd web_app
python run.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload a video file by either:
   - Dragging and dropping the file onto the upload area
   - Clicking "Choose File" and selecting the file

2. Wait for the processing to complete
   - Progress will be displayed in real-time
   - Processing time depends on video length and hardware capabilities

3. View Results
   - The processed video will be displayed
   - Position predictions will be shown in a table format
   - Timestamps will be included for each prediction

## Project Structure

```
project490/
├── web_app/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── models.py
│   │   └── templates/
│   │       └── index.html
│   └── run.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

