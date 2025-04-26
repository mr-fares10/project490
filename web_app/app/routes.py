from flask import render_template, request, jsonify, send_file, send_from_directory, Blueprint
from app import app
import os
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from pathlib import Path
import time
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import queue
import threading
from datetime import datetime, timedelta
import traceback
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
import pandas as pd
from app.video_processor import FIGHTER_JET_DIMENSIONS, AVIATION_CAMERAS
import uuid
import json
from .video_processor import VideoProcessor
from .config import (
    UPLOAD_FOLDER,
    RESULTS_FOLDER,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_MODEL_EXTENSIONS
)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import logging

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variable to track processing progress
processing_status = {
    'current_frame': 0,
    'total_frames': 0,
    'start_time': None,
    'completed': False,
    'fps': 0,
    'status_message': 'Initializing...'
}

# Load the model (you'll need to implement model loading based on your specific model)
model = None

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# LSTM Model Definition
class JetPositionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, num_layers=2, output_size=3):
        super(JetPositionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Create blueprint
bp = Blueprint('main', __name__)

# Store active jobs
active_jobs = {}

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Create directories if they don't exist
UPLOAD_DIR = Path("web_app/uploads")
RESULTS_DIR = Path("web_app/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@bp.route('/')
def index():
    """Render the main page."""
    return render_template('index.html',
                         jet_types=list(FIGHTER_JET_DIMENSIONS.keys()),
                         camera_models=list(AVIATION_CAMERAS.keys()))

@bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Create unique directory for this upload
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save video file
        video_path = os.path.join(upload_dir, secure_filename(video_file.filename))
        video_file.save(video_path)
        
        # Initialize model path as None (will use default)
        model_path = None
        
        # If model file is provided, save it
        if 'model' in request.files:
            model_file = request.files['model']
            if model_file.filename != '':
                model_path = os.path.join(upload_dir, secure_filename(model_file.filename))
                model_file.save(model_path)
        
        # Create results directory
        results_dir = os.path.join(app.config['RESULT_FOLDER'], upload_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize video processor
        processor = VideoProcessor(
            video_path=video_path,
            model_path=model_path,  # Will use default if None
            output_dir=results_dir,
            jet_type=request.form.get('jet_type', "F-16 Fighting Falcon"),
            camera_model=request.form.get('camera_model', "Sony Alpha 7R IV"),
            focal_length=float(request.form.get('focal_length', 200.0))
        )
        
        # Start processing
        processor.process_async(
            progress_callback=lambda p: update_progress(upload_id, p),
            completion_callback=lambda: processing_complete(upload_id),
            error_callback=lambda e: processing_error(upload_id, str(e))
        )
        
        return jsonify({
            'message': 'Processing started',
            'upload_id': upload_id
        }), 202
        
    except Exception as e:
        app.logger.error(f"Error processing upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@bp.route('/status/<job_id>')
def get_status(job_id):
    """Get the status of a processing job."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    processor = job['processor']
    
    response = {
        'status': job['status'],
        'progress': job['progress'],
        'current_frame': processor.current_frame,
        'total_frames': processor.total_frames,
    }
    
    # Add current frame preview if available
    if processor.current_frame_preview:
        response['current_frame_preview'] = os.path.basename(processor.current_frame_preview)
    
    # Add results if processing is complete
    if job['status'] == 'completed':
        results = {
            'main_video': 'output_annotated.mp4',
            'segmentation_video': 'output_segmentation.mp4',
            'trajectory_csv': 'trajectory_data.csv',
            'results_csv': 'results.csv',
            'summary': 'prediction_summary.txt',
            'zip': f'{job_id}_results.zip'
        }
        response['results'] = results
    
    # Add error message if failed
    if job['status'] == 'error':
        response['error'] = job['error']
    
    return jsonify(response)

@bp.route('/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    processor = job['processor']
    
    # Stop processing
    processor.stop()
    
    # Clean up job files
    try:
        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_results_folder = os.path.join(RESULTS_FOLDER, job_id)
        
        if os.path.exists(job_upload_folder):
            for file in os.listdir(job_upload_folder):
                os.remove(os.path.join(job_upload_folder, file))
            os.rmdir(job_upload_folder)
            
        if os.path.exists(job_results_folder):
            for file in os.listdir(job_results_folder):
                os.remove(os.path.join(job_results_folder, file))
            os.rmdir(job_results_folder)
    except Exception as e:
        app.logger.error(f"Error cleaning up job files: {str(e)}")
    
    # Remove job from active jobs
    del active_jobs[job_id]
    
    return jsonify({'status': 'cancelled'})

@bp.route('/view/<job_id>/<path:filename>')
def view_file(job_id, filename):
    """View a file from the results folder."""
    return send_from_directory(os.path.join(RESULTS_FOLDER, job_id), filename)

@bp.route('/download/<job_id>/<path:filename>')
def download_file(job_id, filename):
    """Download a file from the results folder."""
    return send_from_directory(
        os.path.join(RESULTS_FOLDER, job_id),
        filename,
        as_attachment=True
    )

@bp.route('/frames/<job_id>')
def get_frames(job_id):
    """Get list of available frame previews for a job."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    frames_dir = os.path.join(RESULTS_FOLDER, job_id, 'frames')
    if not os.path.exists(frames_dir):
        return jsonify({'frames': []})
    
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    return jsonify({'frames': frames})

def update_job_progress(job_id, progress):
    """Update the progress of a job."""
    if job_id in active_jobs:
        active_jobs[job_id]['progress'] = progress

def complete_job(job_id):
    """Mark a job as completed."""
    if job_id in active_jobs:
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100

def fail_job(job_id, error_message):
    """Mark a job as failed."""
    if job_id in active_jobs:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['error'] = error_message

def allowed_file(filename, allowed_extensions):
    """Check if a filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/progress')
def get_progress():
    global processing_status
    
    if not processing_status['total_frames']:
        return jsonify({
            'progress': 0,
            'status_message': 'Waiting to start...',
            'fps': 0,
            'estimated_time': 'Unknown',
            'completed': False
        })
    
    progress = int((processing_status['current_frame'] / processing_status['total_frames']) * 100)
    
    # Calculate estimated time remaining
    if processing_status['start_time'] and processing_status['fps'] > 0:
        elapsed_time = time.time() - processing_status['start_time']
        frames_remaining = processing_status['total_frames'] - processing_status['current_frame']
        estimated_seconds = frames_remaining / processing_status['fps']
        estimated_time = str(timedelta(seconds=int(estimated_seconds)))
    else:
        estimated_time = 'Calculating...'
    
    return jsonify({
        'progress': progress,
        'status_message': processing_status['status_message'],
        'fps': processing_status['fps'],
        'estimated_time': estimated_time,
        'completed': processing_status['completed']
    })

def process_frames_batch(frames, model, transform):
    """Process a batch of frames"""
    batch_tensor = torch.stack([transform(Image.fromarray(frame)) for frame in frames])
    batch_tensor = batch_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
    
    return outputs

def process_video(video_path):
    """
    Process the video and return predictions.
    """
    try:
        # Create output directory
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if video file exists and is readable
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
            
        # Load the LSTM model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'jet_lstm_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print(f"Loading model from {model_path}")
        
        # Add MinMaxScaler to safe globals for PyTorch 2.6+
        torch.serialization.add_safe_globals([MinMaxScaler])
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Warning: Could not load with weights_only=True, trying weights_only=False: {str(e)}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = JetPositionLSTM()
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
            
        # Extract video information
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_video_path = os.path.join(output_dir, f'annotated_{timestamp}.mp4')
        heatmap_video_path = os.path.join(output_dir, f'heatmap_{timestamp}.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        heatmap_out = cv2.VideoWriter(heatmap_video_path, fourcc, fps, (width, height))
        
        # Initialize lists for storing predictions and actual positions
        positions = []
        predictions = []
        timestamps = []
        
        # Initialize processing status
        processing_status.update({
            'current_frame': 0,
            'total_frames': total_frames,
            'start_time': time.time(),
            'completed': False,
            'fps': fps,
            'status_message': 'Starting video processing...'
        })
        
        # Process frames
        frame_count = 0
        sequence = []
        sequence_length = 10  # Number of frames to use for prediction
        
        print(f"Starting video processing: {total_frames} frames at {fps} fps")
        
        # Create progress bar text template
        progress_bar_width = 50
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update progress
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            
            # Create ASCII progress bar
            filled_length = int(progress_bar_width * frame_count // total_frames)
            bar = '=' * filled_length + '-' * (progress_bar_width - filled_length)
            
            # Calculate processing speed and estimated time remaining
            elapsed_time = time.time() - processing_status['start_time']
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            remaining_frames = total_frames - frame_count
            eta = remaining_frames / current_fps if current_fps > 0 else 0
            
            # Update status message
            status_msg = f'Frame {frame_count}/{total_frames} [{bar}] {progress:.1f}%'
            status_msg += f' | {current_fps:.1f} fps | ETA: {timedelta(seconds=int(eta))}'
            
            processing_status.update({
                'current_frame': frame_count,
                'fps': current_fps,
                'status_message': status_msg
            })
            
            if frame_count % 10 == 0:
                print(status_msg)
            
            # Convert frame to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame for object detection (placeholder)
            x = width / 2
            y = height / 2
            z = 0  # Placeholder depth
            
            current_pos = [x, y, z]
            positions.append(current_pos)
            sequence.append(current_pos)
            
            # Create copies of the frame for annotation
            annotated_frame = frame.copy()
            
            # Generate heatmap visualization
            heatmap_frame = process_frame_heatmap(frame)
            
            # Once we have enough frames in the sequence
            if len(sequence) >= sequence_length:
                # Convert sequence to tensor
                seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    prediction = model(seq_tensor)
                
                # Store prediction
                pred_pos = prediction[0].cpu().numpy().tolist()
                predictions.append(pred_pos)
                timestamps.append(frame_count / fps)
                
                # Draw actual and predicted positions
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # Actual (blue)
                cv2.circle(annotated_frame, (int(pred_pos[0]), int(pred_pos[1])), 5, (0, 0, 255), -1)  # Predicted (red)
                
                # Calculate prediction error
                error = np.sqrt(sum((np.array(current_pos) - np.array(pred_pos))**2))
                
                # Add frame information and legend
                info_text = [
                    f'Frame: {frame_count}/{total_frames}',
                    f'Time: {frame_count/fps:.2f}s',
                    f'Error: {error:.2f}m',
                    'Blue: Actual Position',
                    'Red: Predicted Position'
                ]
                
                for i, text in enumerate(info_text):
                    y_pos = 30 + (i * 25)
                    cv2.putText(annotated_frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Update sequence
                sequence = sequence[1:]
            else:
                # If we don't have enough frames yet, just show actual position
                cv2.circle(annotated_frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                
                # Add frame information
                info_text = [
                    f'Frame: {frame_count}/{total_frames}',
                    f'Time: {frame_count/fps:.2f}s',
                    'Collecting initial sequence...',
                    f'Frames needed: {sequence_length-len(sequence)} more'
                ]
                
                for i, text in enumerate(info_text):
                    y_pos = 30 + (i * 25)
                    cv2.putText(annotated_frame, text, (10, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame counter to heatmap
            cv2.putText(heatmap_frame, f'Frame: {frame_count}/{total_frames}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(heatmap_frame, f'Time: {frame_count/fps:.2f}s', (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frames to output videos
            out.write(annotated_frame)
            heatmap_out.write(heatmap_frame)
        
        # Cleanup
        cap.release()
        out.release()
        heatmap_out.release()
        
        # Update processing status
        processing_status.update({
            'completed': True,
            'status_message': 'Processing completed successfully'
        })
        
        print("Video processing completed")
        
        # Create trajectory plots and save data
        positions_df = pd.DataFrame({
            'frame': range(len(positions)),
            'timestamp': [i/fps for i in range(len(positions))],
            'actual_x': [p[0] for p in positions],
            'actual_y': [p[1] for p in positions],
            'actual_z': [p[2] for p in positions],
            'predicted_x': [p[0] if i < len(predictions) else None for i, p in enumerate(positions)],
            'predicted_y': [p[1] if i < len(predictions) else None for i, p in enumerate(positions)],
            'predicted_z': [p[2] if i < len(predictions) else None for i, p in enumerate(positions)]
        })
        
        # Save trajectory data
        csv_path = os.path.join(output_dir, f'trajectory_data_{timestamp}.csv')
        positions_df.to_csv(csv_path, index=False)
        
        # Create trajectory visualizations
        create_trajectory_comparison(positions_df, output_dir)
        
        # Get relative paths for the output files
        output_video_rel = os.path.relpath(output_video_path, app.config['UPLOAD_FOLDER'])
        heatmap_video_rel = os.path.relpath(heatmap_video_path, app.config['UPLOAD_FOLDER'])
        trajectory_3d_rel = os.path.join('results', 'trajectory_comparison_3d.png')
        trajectory_2d_rel = os.path.join('results', 'trajectory_comparison_2d.png')
        csv_rel = os.path.relpath(csv_path, app.config['UPLOAD_FOLDER'])
        
        return {
            'success': True,
            'message': 'Video processed successfully',
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps > 0 else 0
            },
            'output_files': {
                'main_video_url': output_video_rel,
                'heatmap_video_url': heatmap_video_rel,
                'trajectory_3d_url': trajectory_3d_rel,
                'trajectory_2d_url': trajectory_2d_rel,
                'trajectory_csv_url': csv_rel
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        processing_status.update({
            'status_message': f'Error: {str(e)}',
            'completed': True
        })
        raise Exception(f"Error processing video: {str(e)}")

def create_trajectory_comparison(positions_df, output_dir):
    """
    Create visualizations comparing actual and predicted trajectories
    """
    if not all(col in positions_df.columns for col in ["predicted_x", "predicted_y", "predicted_z"]):
        print("No prediction data available for trajectory comparison")
        return
    
    # Create 3D trajectory plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Drop rows with missing predictions
    df = positions_df.dropna(subset=['predicted_x', 'predicted_y', 'predicted_z'])
    
    if len(df) < 2:
        print("Not enough prediction data for trajectory visualization")
        return
    
    # 3D Trajectory
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot actual trajectory
    ax.plot(df['actual_x'], df['actual_y'], df['actual_z'], 'b-', linewidth=2, label='Actual')
    ax.scatter(df['actual_x'], df['actual_y'], df['actual_z'], c='b', s=30)
    
    # Plot predicted trajectory
    ax.plot(df['predicted_x'], df['predicted_y'], df['predicted_z'], 'r--', linewidth=2, label='Predicted')
    ax.scatter(df['predicted_x'], df['predicted_y'], df['predicted_z'], c='r', s=30)
    
    # Add labels and legend
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Actual vs Predicted 3D Trajectory')
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2D views
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Top-down view (X-Y)
    axs[0].plot(df['actual_x'], df['actual_y'], 'b-', label='Actual')
    axs[0].plot(df['predicted_x'], df['predicted_y'], 'r--', label='Predicted')
    axs[0].set_xlabel('X (meters)')
    axs[0].set_ylabel('Y (meters)')
    axs[0].set_title('Top-Down View (X-Y)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Side view (X-Z)
    axs[1].plot(df['actual_x'], df['actual_z'], 'b-', label='Actual')
    axs[1].plot(df['predicted_x'], df['predicted_z'], 'r--', label='Predicted')
    axs[1].set_xlabel('X (meters)')
    axs[1].set_ylabel('Z (meters)')
    axs[1].set_title('Side View (X-Z)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Side view (Y-Z)
    axs[2].plot(df['actual_y'], df['actual_z'], 'b-', label='Actual')
    axs[2].plot(df['predicted_y'], df['predicted_z'], 'r--', label='Predicted')
    axs[2].set_xlabel('Y (meters)')
    axs[2].set_ylabel('Z (meters)')
    axs[2].set_title('Side View (Y-Z)')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.exception_handler(404)
async def not_found_error(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Not found"}
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Create job-specific directories
        job_upload_dir = UPLOAD_DIR / job_id
        job_results_dir = RESULTS_DIR / job_id
        job_upload_dir.mkdir(parents=True)
        job_results_dir.mkdir(parents=True)
        
        # Save uploaded video
        video_path = job_upload_dir / file.filename
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Initialize video processor
        processor = VideoProcessor(
            str(video_path),
            str(job_results_dir),
            model_path="web_app/app/models/jet_lstm_model.pt"
        )
        
        # Process video asynchronously
        processor.process_async(
            progress_callback=lambda progress: logger.info(f"Processing progress: {progress}%")
        )
        
        return JSONResponse({
            "status": "success",
            "message": "Video upload successful",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/process-status/{job_id}")
async def get_process_status(job_id: str):
    try:
        results_dir = RESULTS_DIR / job_id
        if not results_dir.exists():
            return JSONResponse({
                "status": "processing",
                "progress": 0
            })
            
        # Check for completed files
        output_video = results_dir / "output_annotated.mp4"
        if output_video.exists():
            return JSONResponse({
                "status": "completed",
                "progress": 100,
                "output_url": f"/api/results/{job_id}/output_annotated.mp4"
            })
            
        return JSONResponse({
            "status": "processing",
            "progress": 50  # You might want to implement more precise progress tracking
        })
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{job_id}/{filename}")
async def get_result_file(job_id: str, filename: str):
    try:
        file_path = RESULTS_DIR / job_id / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        return FileResponse(str(file_path))
        
    except Exception as e:
        logger.error(f"Error serving result file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
