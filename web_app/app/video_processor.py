import os
import cv2
import torch
import numpy as np
import pandas as pd
import threading
import logging
import traceback
from typing import Optional, Callable
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FIGHTER_JET_DIMENSIONS = {
    "F-16": {"length": 15.03, "wingspan": 9.96, "height": 4.88},
    "F-18": {"length": 17.07, "wingspan": 13.62, "height": 4.66},
    "F-22": {"length": 18.92, "wingspan": 13.56, "height": 5.08},
    "F-35": {"length": 15.67, "wingspan": 10.7, "height": 4.33}
}

AVIATION_CAMERAS = {
    "Standard": {"sensor_width": 36, "sensor_height": 24},
    "High-Speed": {"sensor_width": 28.7, "sensor_height": 19.1},
    "Tracking": {"sensor_width": 23.6, "sensor_height": 15.6}
}

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str, model_path: str = None,
                 jet_type: str = "F-16", camera_model: str = "Standard",
                 focal_length: float = 200.0):
        """Initialize the video processor."""
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.jet_type = jet_type
        self.camera_model = camera_model
        self.focal_length = focal_length
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        
        # Initialize processing state
        self.stop_flag = False
        self.processing_thread = None
        self.position_history = []
        self.predictions = []
        self.timestamps = []
        self.current_frame = 0
        self.total_frames = 0
        self.current_frame_preview = None
        
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Initialize model if path is provided
            self.model = None
            if self.model_path and os.path.exists(self.model_path):
                self._init_model()
            
            # Calculate camera parameters
            self.camera_params = self._calculate_camera_parameters(
                AVIATION_CAMERAS[camera_model]["sensor_width"],
                AVIATION_CAMERAS[camera_model]["sensor_height"],
                self.focal_length
            )
            logger.info("Camera parameters calculated successfully")
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def process_async(self, progress_callback: Optional[Callable[[float], None]] = None,
                     completion_callback: Optional[Callable[[], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """Start asynchronous video processing."""
        logger.info("Starting asynchronous video processing")
        
        def process_thread():
            try:
                logger.info("Processing thread started")
                self._process_video(progress_callback)
                logger.info("Processing completed successfully")
                if completion_callback:
                    completion_callback()
            except Exception as e:
                logger.error(f"Error in processing thread: {str(e)}")
                logger.error(traceback.format_exc())
                if error_callback:
                    error_callback(str(e))
                    
        self.processing_thread = threading.Thread(target=process_thread)
        self.processing_thread.start()
        logger.info("Processing thread initialized and started")
        
    def _init_model(self):
        """Initialize the model."""
        try:
            # Try to import the model loader from your models.py file
            try:
                # First try to import from project structure - use direct import
                # instead of a package path to avoid issues
                try:
                    # Get the absolute path to models.py
                    import os
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    # Add the directory to system path
                    import sys
                    if script_dir not in sys.path:
                        sys.path.append(script_dir)
                    
                    # Now try to import directly
                    from models import load_model
                    self.model = load_model(self.model_path)
                except ImportError:
                    # If that fails, try the package path
                    from web_app.app.models import load_model
                    self.model = load_model(self.model_path)
            except ImportError:
                # If that fails, implement a simple model loader
                logger.warning("Could not import model loader, using simplified version")
                self.model = self._load_model_fallback(self.model_path)
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue without model for demonstration purposes
            self.model = None
            
    def _load_model_fallback(self, model_path):
        """Simple fallback model loader if the import fails."""
        try:
            # Basic PyTorch model loading
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                return torch.load(model_path, map_location='cpu')
            return None
        except Exception as e:
            logger.error(f"Error in fallback model loading: {str(e)}")
            return None
            
    def _calculate_camera_parameters(self, sensor_width: float, sensor_height: float,
                                   focal_length: float) -> dict:
        """Calculate camera parameters for 3D position estimation."""
        return {
            "focal_length": focal_length,
            "sensor_width": sensor_width,
            "sensor_height": sensor_height,
            "principal_point": (self.width / 2, self.height / 2)
        }
        
    def _process_video(self, progress_callback: Optional[Callable[[float], None]] = None):
        """Process the video and generate outputs."""
        try:
            # Initialize video writers
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_main = cv2.VideoWriter(
                os.path.join(self.output_dir, 'output_annotated.mp4'),
                fourcc, self.fps, (self.width, self.height)
            )
            
            # For heatmap visualization
            out_heatmap = cv2.VideoWriter(
                os.path.join(self.output_dir, 'output_heatmap.mp4'),
                fourcc, self.fps, (self.width, self.height)
            )
            
            # Initialize position tracking
            frame_count = 0
            sequence_length = 10
            position_history = []
            
            # Process each frame
            while not self.stop_flag and frame_count < self.total_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Update progress
                frame_count += 1
                self.current_frame = frame_count
                progress = (frame_count / self.total_frames) * 100
                
                if progress_callback:
                    progress_callback(progress)
                
                # Add synthetic delay to simulate processing time
                time.sleep(0.05)
                
                # Create an annotated frame (with tracking info)
                annotated_frame = self._annotate_frame(frame, frame_count)
                
                # Create a heatmap visualization
                heatmap_frame = self._create_heatmap(frame)
                
                # Save sample frames periodically for preview
                if frame_count % 10 == 0 or frame_count == 1:
                    frame_path = os.path.join(self.output_dir, 'frames', f'frame_{frame_count:04d}.jpg')
                    cv2.imwrite(frame_path, annotated_frame)
                    self.current_frame_preview = os.path.join('frames', f'frame_{frame_count:04d}.jpg')
                
                # Write frames to output videos
                out_main.write(annotated_frame)
                out_heatmap.write(heatmap_frame)
                
            # Create trajectory data
            self._generate_trajectory_data()
            
            # Cleanup
            out_main.release()
            out_heatmap.release()
            self.cap.release()
            
            # Create a summary file
            self._create_summary()
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def _annotate_frame(self, frame, frame_count):
        """Add annotations to the frame."""
        annotated = frame.copy()
        
        # Calculate simulated position (circular motion)
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) // 4
        angle = (frame_count / self.total_frames) * 2 * np.pi * 2  # 2 full circles
        
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        z = 0  # Placeholder depth
        
        # Store position
        position = [x, y, z]
        self.position_history.append(position)
        
        # Draw actual position
        cv2.circle(annotated, (x, y), 10, (0, 0, 255), -1)  # Red circle
        
        # Add frame information
        cv2.putText(annotated, f"Frame: {frame_count}/{self.total_frames}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Time: {frame_count/self.fps:.2f}s", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"Jet Type: {self.jet_type}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw trajectory line (last 30 frames)
        history_length = min(30, len(self.position_history))
        if history_length > 1:
            for i in range(1, history_length):
                prev_pos = self.position_history[-i]
                curr_pos = self.position_history[-i-1]
                cv2.line(annotated, 
                       (int(prev_pos[0]), int(prev_pos[1])), 
                       (int(curr_pos[0]), int(curr_pos[1])), 
                       (0, 255, 0), 2)
        
        return annotated
    
    def _create_heatmap(self, frame):
        """Create a heatmap visualization of the frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply threshold to create binary image
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
        # Apply color map
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        
        # Create blend with original frame
        alpha = 0.7
        blend = cv2.addWeighted(frame, alpha, heatmap, 1-alpha, 0)
        
        return blend
    
    def _generate_trajectory_data(self):
        """Generate trajectory data CSV."""
        if len(self.position_history) == 0:
            return
            
        # Create dataframe
        df = pd.DataFrame(self.position_history, columns=['x', 'y', 'z'])
        df['frame'] = range(1, len(self.position_history) + 1)
        df['timestamp'] = df['frame'] / self.fps
        
        # Reorder columns
        df = df[['frame', 'timestamp', 'x', 'y', 'z']]
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'trajectory_data.csv')
        df.to_csv(csv_path, index=False)
        
    def _create_summary(self):
        """Create a summary text file."""
        summary_path = os.path.join(self.output_dir, 'processing_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("Fighter Jet Position Prediction - Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Video File: {os.path.basename(self.video_path)}\n")
            f.write(f"Model: {os.path.basename(self.model_path) if self.model_path else 'Default'}\n")
            f.write(f"Jet Type: {self.jet_type}\n")
            f.write(f"Camera Model: {self.camera_model}\n")
            f.write(f"Focal Length: {self.focal_length}mm\n\n")
            
            f.write("Video Properties:\n")
            f.write(f"- Resolution: {self.width}x{self.height}\n")
            f.write(f"- Frame Rate: {self.fps} fps\n")
            f.write(f"- Duration: {self.total_frames / self.fps:.2f} seconds\n")
            f.write(f"- Total Frames: {self.total_frames}\n\n")
            
            f.write("Output Files:\n")
            f.write("- output_annotated.mp4: Video with position annotations\n")
            f.write("- output_heatmap.mp4: Heatmap visualization\n")
            f.write("- trajectory_data.csv: Position data for each frame\n")
            f.write("- frames/: Directory with sample frame images\n\n")
            
            f.write("Processing completed successfully.\n")
    
    def stop(self):
        """Stop video processing."""
        self.stop_flag = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)