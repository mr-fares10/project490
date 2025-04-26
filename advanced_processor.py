import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
import logging
import threading
from typing import Optional, Callable
import pandas as pd
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced-processor")

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Dictionary of fighter jet reference dimensions (in meters)
FIGHTER_JET_DIMENSIONS = {
    "F-16": {
        "length": 15.06,
        "wingspan": 9.96,
        "height": 5.09
    },
    "F-18": {
        "length": 17.07,
        "wingspan": 13.62,
        "height": 4.66
    },
    "F-22": {
        "length": 18.92,
        "wingspan": 13.56,
        "height": 5.08
    },
    "F-35": {
        "length": 15.67,
        "wingspan": 10.7,
        "height": 4.38
    }
}

# Dictionary of aviation camera specifications
AVIATION_CAMERAS = {
    "Standard": {
        "sensor_width": 35.7,  # mm
        "sensor_height": 23.8,  # mm
        "resolution": (1920, 1080),  # pixels
        "pixel_size": 0.00375,  # mm
        "typical_focal_length": 85  # mm
    },
    "High-Speed": {
        "sensor_width": 35.7,  # mm
        "sensor_height": 23.8,  # mm
        "resolution": (1920, 1080),  # pixels
        "pixel_size": 0.00375,  # mm
        "typical_focal_length": 200  # mm
    },
    "Tracking": {
        "sensor_width": 35.7,  # mm
        "sensor_height": 23.8,  # mm
        "resolution": (1920, 1080),  # pixels
        "pixel_size": 0.00375,  # mm
        "typical_focal_length": 300  # mm
    }
}

# LSTM Model Definition
class JetPositionLSTM(nn.Module):
    """LSTM model for predicting the next position of a fighter jet"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(JetPositionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class AdvancedVideoProcessor:
    """Advanced video processor that uses object detection and LSTM prediction"""
    
    def __init__(self, video_path: str, output_dir: str, model_path: str = None,
                 jet_type: str = "F-16", camera_model: str = "Standard",
                 focal_length: float = 200.0):
        """Initialize the video processor"""
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
            
            logger.info(f"Video loaded: {video_path}")
            logger.info(f"Properties: {self.width}x{self.height}, {self.fps} fps, {self.total_frames} frames")
            
            # Calculate camera parameters
            self.camera_params = self._calculate_camera_parameters()
            
            # Initialize LSTM model
            self._init_model()
            
            # Initialize detection models
            self._init_detection_models()
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_camera_parameters(self):
        """Calculate camera intrinsic parameters"""
        camera_specs = AVIATION_CAMERAS.get(self.camera_model, AVIATION_CAMERAS["Standard"])
        
        # Use provided focal length
        actual_focal_length = self.focal_length
        
        # Calculate scaling factor
        width_scale = self.width / camera_specs["resolution"][0]
        height_scale = self.height / camera_specs["resolution"][1]
        
        # Convert focal length from mm to pixels
        focal_length_x_pixels = actual_focal_length / camera_specs["pixel_size"] * width_scale
        focal_length_y_pixels = actual_focal_length / camera_specs["pixel_size"] * height_scale
        
        # Principal point (center of image)
        cx = self.width / 2
        cy = self.height / 2
        
        return {
            "fx": focal_length_x_pixels,
            "fy": focal_length_y_pixels,
            "cx": cx,
            "cy": cy,
            "focal_length_mm": actual_focal_length
        }
    
    def _init_model(self):
        """Initialize the LSTM model"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading LSTM model from {self.model_path}")
                
                # Load the model
                checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
                
                # Check if it's a dictionary with model info
                if isinstance(checkpoint, dict) and 'model_info' in checkpoint:
                    # Get model architecture from saved info
                    hidden_size = checkpoint['model_info']['hidden_size']
                    num_layers = checkpoint['model_info']['num_layers']
                    
                    # Create model with saved architecture
                    self.lstm_model = JetPositionLSTM(
                        input_size=3,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        output_size=3
                    ).to(device)
                    
                    self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                    self.scaler = checkpoint['scaler']
                    self.sequence_length = checkpoint['sequence_length']
                    
                    logger.info(f"Model loaded successfully. Sequence length: {self.sequence_length}")
                    
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Create model with default architecture
                    self.lstm_model = JetPositionLSTM().to(device)
                    self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
                    self.sequence_length = 10
                    
                    logger.info("Model loaded with partial info - using default sequence length of 10")
                    
                else:
                    # Assume it's just the state dict
                    self.lstm_model = JetPositionLSTM().to(device)
                    self.lstm_model.load_state_dict(checkpoint)
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
                    self.sequence_length = 10
                    
                    logger.info("Model loaded with state dict only - using default parameters")
                
                self.lstm_model.eval()
                
            else:
                logger.warning(f"No valid model path provided: {self.model_path}")
                self.lstm_model = None
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                self.sequence_length = 10
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            self.lstm_model = None
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.sequence_length = 10
    
    def _init_detection_models(self):
        """Initialize object detection models"""
        try:
            # Load segmentation model
            import torchvision
            from torchvision import transforms
            
            logger.info("Loading Mask R-CNN model...")
            self.seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            self.seg_model.to(device)
            self.seg_model.eval()
            
            try:
                # Try to load MiDaS depth model
                logger.info("Loading MiDaS depth model...")
                self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
                self.depth_model.to(device)
                self.depth_model.eval()
                self.has_depth_model = True
                
                # Define transforms for depth model
                self.depth_transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
            except Exception as e:
                logger.warning(f"Error loading depth model: {str(e)}")
                self.has_depth_model = False
            
            # Define transforms for segmentation model
            self.transform = transforms.ToTensor()
            
            logger.info("Detection models loaded successfully!")
            self.has_detection = True
            
        except Exception as e:
            logger.error(f"Error loading detection models: {str(e)}")
            logger.error(traceback.format_exc())
            self.has_detection = False
    
    def process_async(self, progress_callback: Optional[Callable[[float], None]] = None,
                     completion_callback: Optional[Callable[[], None]] = None,
                     error_callback: Optional[Callable[[str], None]] = None):
        """Start asynchronous video processing"""
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
    
    def _process_video(self, progress_callback: Optional[Callable[[float], None]] = None):
        """Process the video and generate outputs"""
        try:
            # Initialize video writers
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Main output video with annotations
            out_main = cv2.VideoWriter(
                os.path.join(self.output_dir, 'output_annotated.mp4'),
                fourcc, self.fps, (self.width, self.height)
            )
            
            # Segmentation video with predictions
            out_seg = cv2.VideoWriter(
                os.path.join(self.output_dir, 'segmentation_with_predictions.mp4'),
                fourcc, self.fps, (self.width, self.height)
            )
            
            # Create trajectory data file
            trajectory_file = os.path.join(self.output_dir, 'trajectory_with_predictions.csv')
            with open(trajectory_file, 'w') as f:
                f.write("frame,timestamp,actual_x,actual_y,actual_z,predicted_x,predicted_y,predicted_z,error_3d\n")
            
            # Initialize lists for tracking
            position_history = []
            position_data = []
            
            # Process frames
            frame_count = 0
            
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
                
                # Process current frame
                logger.info(f"Processing frame {frame_count}/{self.total_frames}")
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get camera parameters
                fx, fy = self.camera_params["fx"], self.camera_params["fy"]
                cx, cy = self.camera_params["cx"], self.camera_params["cy"]
                
                # Create a copy for visualization
                annotated_frame = frame.copy()
                segmentation = frame_rgb.copy()
                
                # Track aircraft position
                try:
                    if self.has_detection:
                        # Use the detection model to find aircraft
                        position_found, position_data_row = self._detect_aircraft(
                            frame_rgb, frame_count, fx, fy, cx, cy
                        )
                        
                        if position_found:
                            # Get the actual position
                            x_meters = position_data_row["actual_x"]
                            y_meters = position_data_row["actual_y"]
                            z_meters = position_data_row["actual_z"]
                            
                            # Process for visualization
                            segmentation = position_data_row["segmentation"]
                            
                            # Add to position history and data
                            position_history.append([x_meters, y_meters, z_meters])
                            position_data.append(position_data_row)
                            
                            # Write to trajectory file
                            with open(trajectory_file, 'a') as f:
                                if "predicted_x" in position_data_row:
                                    pred_x = position_data_row["predicted_x"]
                                    pred_y = position_data_row["predicted_y"]
                                    pred_z = position_data_row["predicted_z"]
                                    error_3d = position_data_row["error_3d"]
                                    
                                    f.write(f"{frame_count},{frame_count/self.fps:.3f}," +
                                           f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f}," +
                                           f"{pred_x:.3f},{pred_y:.3f},{pred_z:.3f},{error_3d:.3f}\n")
                                else:
                                    f.write(f"{frame_count},{frame_count/self.fps:.3f}," +
                                           f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f},,,,\n")
                        else:
                            # If aircraft not detected, use simulated position for demo purposes
                            self._simulate_aircraft_position(segmentation, frame_count, trajectory_file)
                    else:
                        # No detection model, use simulated position
                        self._simulate_aircraft_position(segmentation, frame_count, trajectory_file)
                        
                except Exception as e:
                    logger.error(f"Error in aircraft detection: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue processing without this frame's detection
                
                # Convert RGB to BGR for OpenCV
                segmentation_bgr = cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR)
                
                # Write to videos
                out_main.write(frame)  # Original frame
                out_seg.write(segmentation_bgr)  # Segmentation with predictions
                
                # Save sample frames
                if frame_count % 10 == 0 or frame_count == 1:
                    frame_path = os.path.join(self.output_dir, 'frames', f'frame_{frame_count:04d}.jpg')
                    cv2.imwrite(frame_path, segmentation_bgr)
                    self.current_frame_preview = os.path.join('frames', f'frame_{frame_count:04d}.jpg')
            
            # Close videos
            out_main.release()
            out_seg.release()
            
            # Create dataframe of results
            if position_data:
                results_df = pd.DataFrame(position_data)
                results_path = os.path.join(self.output_dir, 'position_prediction_results.csv')
                results_df.to_csv(results_path, index=False)
                
                # Create visualizations if we have predictions
                if 'predicted_x' in results_df.columns:
                    self._create_trajectory_comparison(results_df)
            
            # Create summary file
            self._create_summary(position_data)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _detect_aircraft(self, frame_rgb, frame_count, fx, fy, cx, cy):
        """Detect aircraft in the frame and get its position"""
        try:
            # Convert to PIL image for transforms
            pil_image = Image.fromarray(frame_rgb)
            
            # Process with segmentation model
            seg_tensor = self.transform(frame_rgb).to(device)
            
            with torch.no_grad():
                prediction = self.seg_model([seg_tensor])
            
            # Check if any objects detected
            scores = prediction[0]['scores']
            
            if len(scores) > 0:
                # Filter high confidence detections
                high_conf_indices = torch.where(scores > 0.5)[0]
                
                if len(high_conf_indices) > 0:
                    # Look for airplane class (index 4 in COCO)
                    airplane_indices = [i for i in high_conf_indices if prediction[0]['labels'][i].item() == 4]
                    
                    if airplane_indices:
                        best_idx = airplane_indices[0].item()
                    else:
                        # If no airplane, use highest confidence object
                        best_idx = high_conf_indices[0].item()
                    
                    # Get mask and class
                    mask = prediction[0]['masks'][best_idx, 0].cpu().numpy()
                    score = scores[best_idx].item()
                    label = prediction[0]['labels'][best_idx].item()
                    
                    # Threshold mask
                    binary_mask = mask > 0.5
                    y_indices, x_indices = np.where(binary_mask)
                    
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        # Compute 2D centroid
                        u_centroid = np.mean(x_indices)
                        v_centroid = np.mean(y_indices)
                        
                        # Get bounding box dimensions
                        rect = cv2.minAreaRect(np.column_stack((x_indices, y_indices)))
                        box = cv2.boxPoints(rect)
                        box = box.astype(np.int32)
                        
                        rect_width, rect_height = rect[1]
                        
                        # Ensure width is the longer dimension
                        if rect_width < rect_height:
                            rect_width, rect_height = rect_height, rect_width
                        
                        # Get depth estimation if available
                        aircraft_depth_meters = None
                        
                        if self.has_depth_model:
                            # Process with depth model
                            depth_input = self.depth_transform(pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                depth_pred = self.depth_model(depth_input)
                            
                            depth_map = depth_pred.squeeze().cpu().numpy()
                            
                            # Resize for calculation
                            depth_map_resized = cv2.resize(depth_map, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                            aircraft_relative_depth = depth_map_resized[y_indices, x_indices].mean()
                        
                        # Distance calculation using reference dimensions
                        jet_specs = FIGHTER_JET_DIMENSIONS.get(self.jet_type, FIGHTER_JET_DIMENSIONS["F-16"])
                        reference_length = jet_specs["length"]
                        aircraft_depth_meters = (reference_length * fx) / rect_width
                        
                        # Convert 2D centroid to 3D coordinates
                        x_meters = (u_centroid - cx) * aircraft_depth_meters / fx
                        y_meters = (v_centroid - cy) * aircraft_depth_meters / fy
                        z_meters = aircraft_depth_meters
                        
                        # Current actual position
                        actual_position = [x_meters, y_meters, z_meters]
                        
                        # Add to position history for LSTM
                        self.position_history.append(actual_position)
                        
                        # Prepare visualization
                        segmentation = frame_rgb.copy()
                        
                        # Draw mask overlay
                        mask_overlay = np.zeros_like(segmentation)
                        mask_overlay[binary_mask, 0] = 255  # Red channel
                        segmentation = cv2.addWeighted(segmentation, 1, mask_overlay, 0.5, 0)
                        
                        # Draw actual position (green dot)
                        cv2.circle(segmentation, (int(u_centroid), int(v_centroid)), 5, (0, 255, 0), -1)
                        
                        # Draw oriented bounding box
                        cv2.drawContours(segmentation, [box], 0, (0, 255, 255), 2)
                        
                        # Add position text
                        cv2.putText(segmentation,
                                  f"Pos (m): X={x_meters:.2f}, Y={y_meters:.2f}, Z={z_meters:.2f}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Results row for this frame
                        position_row = {
                            "frame": frame_count,
                            "timestamp": frame_count/self.fps,
                            "actual_x": x_meters,
                            "actual_y": y_meters,
                            "actual_z": z_meters,
                            "segmentation": segmentation
                        }
                        
                        # If we have enough position history, make prediction with the LSTM model
                        if self.lstm_model is not None and len(self.position_history) >= self.sequence_length:
                            # Get the most recent positions
                            recent_positions = self.position_history[-self.sequence_length:]
                            
                            try:
                                # Scale data for LSTM
                                try:
                                    # Try with existing scaler
                                    scaled_positions = self.scaler.transform(recent_positions)
                                except:
                                    # If that fails, fit a new scaler
                                    logger.warning("Rescaling data with a new scaler")
                                    self.scaler = MinMaxScaler(feature_range=(-1, 1))
                                    self.scaler.fit(recent_positions)
                                    scaled_positions = self.scaler.transform(recent_positions)
                                
                                # Convert to tensor
                                X = torch.FloatTensor(scaled_positions).unsqueeze(0).to(device)
                                
                                # Make prediction
                                with torch.no_grad():
                                    scaled_prediction = self.lstm_model(X).cpu().numpy()
                                
                                # Convert back to original scale
                                predicted_position = self.scaler.inverse_transform(scaled_prediction)[0]
                                pred_x, pred_y, pred_z = predicted_position
                                
                                # Calculate error
                                error_3d = np.sqrt((pred_x - x_meters)**2 + (pred_y - y_meters)**2 + (pred_z - z_meters)**2)
                                
                                # Project predicted position to 2D for visualization
                                pred_u = fx * pred_x / pred_z + cx
                                pred_v = fy * pred_y / pred_z + cy
                                
                                # Draw prediction (violet dot)
                                cv2.circle(segmentation, (int(pred_u), int(pred_v)), 8, (255, 0, 255), -1)
                                cv2.circle(segmentation, (int(pred_u), int(pred_v)), 8, (0, 0, 0), 2)  # Black outline
                                
                                # Draw line connecting actual and predicted
                                cv2.line(segmentation,
                                        (int(u_centroid), int(v_centroid)),
                                        (int(pred_u), int(pred_v)),
                                        (255, 255, 0), 2)  # Yellow line
                                
                                # Add prediction info
                                cv2.putText(segmentation,
                                          f"Pred: X={pred_x:.2f}, Y={pred_y:.2f}, Z={pred_z:.2f}",
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                
                                cv2.putText(segmentation,
                                          f"Error: {error_3d:.2f}m",
                                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                
                                cv2.putText(segmentation,
                                          "PREDICTION ACTIVE",
                                          (self.width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                
                                # Add prediction to result
                                position_row.update({
                                    "predicted_x": pred_x,
                                    "predicted_y": pred_y,
                                    "predicted_z": pred_z,
                                    "error_3d": error_3d,
                                    "segmentation": segmentation
                                })
                                
                            except Exception as e:
                                logger.error(f"Error in LSTM prediction: {str(e)}")
                                logger.error(traceback.format_exc())
                        else:
                            # Not enough history for prediction yet
                            if self.lstm_model is not None:
                                cv2.putText(segmentation,
                                          f"Collecting data: {len(self.position_history)}/{self.sequence_length}",
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        return True, position_row
            
            # No aircraft detected
            return False, None
                
        except Exception as e:
            logger.error(f"Error in aircraft detection: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None
    
    def _simulate_aircraft_position(self, frame, frame_count, trajectory_file):
        """Simulate aircraft position when detection fails"""
        # Calculate simulated position (circular motion)
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) // 4
        angle = (frame_count / self.total_frames) * 2 * np.pi * 2  # 2 full circles
        
        u = int(center_x + radius * np.cos(angle))
        v = int(center_y + radius * np.sin(angle))
        
        # Convert to 3D coordinates
        fx, fy = self.camera_params["fx"], self.camera_params["fy"]
        cx, cy = self.camera_params["cx"], self.camera_params["cy"]
        
        # Assume a fixed depth of 100 meters
        z_meters = 100.0
        x_meters = (u - cx) * z_meters / fx
        y_meters = (v - cy) * z_meters / fy
        
        # Store position
        self.position_history.append([x_meters, y_meters, z_meters])
        
        # Draw simulated position on frame
        cv2.circle(frame, (u, v), 10, (255, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, "SIMULATED POSITION", (center_x - 100, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Write to trajectory file
        with open(trajectory_file, 'a') as f:
            f.write(f"{frame_count},{frame_count/self.fps:.3f}," +
                   f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f},,,,\n")
    
    def _create_trajectory_comparison(self, positions_df):
        """Create visualizations comparing actual and predicted trajectories"""
        if not all(col in positions_df.columns for col in ["predicted_x", "predicted_y", "predicted_z"]):
            logger.info("No prediction data available for trajectory comparison")
            return

        # Drop rows with missing predictions
        df = positions_df.dropna(subset=['predicted_x', 'predicted_y', 'predicted_z'])

        if len(df) < 2:
            logger.info("Not enough prediction data for trajectory visualization")
            return

        try:
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
            plt.savefig(os.path.join(self.output_dir, 'trajectory_comparison_3d.png'), dpi=300, bbox_inches='tight')
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
            plt.savefig(os.path.join(self.output_dir, 'trajectory_comparison_2d.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Prediction error over time
            if 'error_3d' in df.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(df['frame'], df['error_3d'], 'r-')
                plt.xlabel('Frame')
                plt.ylabel('Prediction Error (meters)')
                plt.title('Prediction Error Over Time')
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'prediction_error.png'), dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("Trajectory comparison visualizations created")
        except Exception as e:
            logger.error(f"Error creating trajectory comparison: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _create_summary(self, position_data):
        """Create a summary of the processing results"""
        summary_path = os.path.join(self.output_dir, 'processing_summary.txt')
        
        try:
            with open(summary_path, 'w') as f:
                f.write("FIGHTER JET LSTM PREDICTION SUMMARY\n")
                f.write("==================================\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video: {os.path.basename(self.video_path)}\n")
                f.write(f"Frames processed: {self.current_frame}\n")
                f.write(f"LSTM model: {os.path.basename(self.model_path) if self.model_path else 'None'}\n\n")
                
                # Check if we have error data
                errors = [row.get('error_3d') for row in position_data if 'error_3d' in row]
                
                if errors:
                    avg_error = sum(errors) / len(errors)
                    max_error = max(errors)
                    min_error = min(errors)
                    
                    f.write("PREDICTION STATISTICS\n")
                    f.write(f"Average error: {avg_error:.3f} meters\n")
                    f.write(f"Maximum error: {max_error:.3f} meters\n")
                    f.write(f"Minimum error: {min_error:.3f} meters\n\n")
                
                f.write("OUTPUT FILES\n")
                f.write("- output_annotated.mp4: Original video\n")
                f.write("- segmentation_with_predictions.mp4: Video with aircraft detection and predictions\n")
                f.write("- trajectory_with_predictions.csv: CSV file with trajectory data\n")
                f.write("- position_prediction_results.csv: Detailed results data\n")
                f.write("- trajectory_comparison_3d.png: 3D visualization of trajectories\n")
                f.write("- trajectory_comparison_2d.png: 2D visualizations of trajectories\n")
                f.write("- prediction_error.png: Graph of prediction error over time\n")
                f.write("- frames/: Directory with sample frame images\n")
                
            logger.info(f"Summary created at {summary_path}")
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
    
    def stop(self):
        """Stop video processing"""
        self.stop_flag = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)


if __name__ == "__main__":
    print("This module provides the AdvancedVideoProcessor class.")
    print("Import it in another script to use it for video processing.")