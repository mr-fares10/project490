# Install required packages
!pip install torch torchvision opencv-python matplotlib scikit-learn pillow

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import torchvision
from torchvision import transforms
from PIL import Image
import json
import pandas as pd
from datetime import datetime
import time
from google.colab import files
import io
from sklearn.preprocessing import MinMaxScaler
import pickle

# LSTM Model Definition (must match the training model architecture)
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

# Load the pre-trained model
def load_lstm_model(model_path):
    """Load a pre-trained LSTM model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    try:
        # Try the less secure approach but more compatible with older models
        print("Loading model with weights_only=False (compatible with older models)")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get model architecture from saved info
        hidden_size = checkpoint['model_info']['hidden_size']
        num_layers = checkpoint['model_info']['num_layers']
        
        # Create model with the saved architecture
        model = JetPositionLSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        
        # Get other necessary components for prediction
        scaler = checkpoint['scaler']
        sequence_length = checkpoint['sequence_length']
        
        print(f"Model loaded successfully from {model_path}")
        print(f"This model was trained on: {checkpoint['model_info']['date_trained']}")
        
        return model, scaler, sequence_length, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        
        # Last resort: handle the case where we might need to manually recreate everything
        try:
            print("Attempting alternative loading method...")
            
            # Create a new model with default parameters
            model = JetPositionLSTM(
                input_size=3,
                hidden_size=64,
                num_layers=2,
                output_size=3
            ).to(device)
            
            # Try to load just the state dict
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Create a new scaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            
            # Use default sequence length
            sequence_length = 10
            
            print("Model loaded with default parameters - predictions may be less accurate")
            return model, scaler, sequence_length, device
            
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            traceback.print_exc()
            raise

# Create trajectory comparison visualizations
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
    
    # Prediction error over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['frame'], df['error_3d'], 'r-')
    plt.xlabel('Frame')
    plt.ylabel('Prediction Error (meters)')
    plt.title('Prediction Error Over Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Trajectory comparison visualizations created")

# Dictionary of fighter jet reference dimensions (in meters)
FIGHTER_JET_DIMENSIONS = {
    "F-16 Fighting Falcon": {
        "length": 15.06,
        "wingspan": 9.96,
        "height": 5.09
    },
    "F/A-18 Hornet": {
        "length": 17.07,
        "wingspan": 13.62,
        "height": 4.66
    },
    "F-22 Raptor": {
        "length": 18.92,
        "wingspan": 13.56,
        "height": 5.08
    },
    "F-35 Lightning II": {
        "length": 15.67,
        "wingspan": 10.7,
        "height": 4.38
    }
}

# Dictionary of aviation camera specifications
AVIATION_CAMERAS = {
    "Sony Alpha 7R IV": {
        "sensor_width": 35.7,  # mm
        "sensor_height": 23.8,  # mm
        "resolution": (9504, 6336),  # pixels
        "pixel_size": 0.00375,  # mm
        "typical_focal_length": 85  # mm
    }
}

# Function to calculate camera parameters
def calculate_camera_parameters(image_shape, camera_model, focal_length_mm=None):
    """Calculate camera intrinsic parameters"""
    camera_specs = AVIATION_CAMERAS.get(camera_model, AVIATION_CAMERAS["Sony Alpha 7R IV"])
    
    # Use provided focal length or typical value
    actual_focal_length = focal_length_mm if focal_length_mm else camera_specs["typical_focal_length"]
    
    # Calculate scaling factor if image resolution differs from camera's native resolution
    width_scale = image_shape[1] / camera_specs["resolution"][0]
    height_scale = image_shape[0] / camera_specs["resolution"][1]
    
    # Convert focal length from mm to pixels
    focal_length_x_pixels = actual_focal_length / camera_specs["pixel_size"] * width_scale
    focal_length_y_pixels = actual_focal_length / camera_specs["pixel_size"] * height_scale
    
    # Principal point (usually center of image)
    cx = image_shape[1] / 2
    cy = image_shape[0] / 2
    
    return {
        "fx": focal_length_x_pixels,
        "fy": focal_length_y_pixels,
        "cx": cx,
        "cy": cy,
        "focal_length_mm": actual_focal_length
    }

# Process video with pre-trained LSTM predictions
def process_video_with_lstm_predictions(video_path, model_path, output_dir="/content/output", 
                                      camera_model="Sony Alpha 7R IV", focal_length_mm=200,
                                      known_jet_type="F-16 Fighting Falcon", fps=60, use_original_fps=True):
    """
    Process a video with the pre-trained LSTM model for position prediction
    
    Parameters:
    - video_path: Path to the video file
    - model_path: Path to the trained LSTM model
    - output_dir: Directory to save results
    - camera_model: Camera model for calculations
    - focal_length_mm: Focal length in mm
    - known_jet_type: Type of fighter jet for reference dimensions
    - fps: Frames per second to process
    - use_original_fps: Use the video's original frame rate
    
    Returns:
    - Dictionary with results and paths to output files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load pre-trained LSTM model
    model, scaler, sequence_length, device = load_lstm_model(model_path)
    
    # Load models for jet detection
    try:
        print("Loading detection models...")
        seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        seg_model.to(device)
        seg_model.eval()
        
        depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        depth_model.to(device)
        depth_model.eval()
        
        print("Detection models loaded successfully!")
    except Exception as e:
        print(f"Error loading detection models: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    print(f"Video loaded: {video_path}")
    print(f"Duration: {duration:.2f} seconds ({total_frames} frames at {video_fps:.2f} fps)")
    
    # Determine the frame rate to use for processing
    if use_original_fps:
        processing_fps = video_fps
        print(f"Using original video frame rate: {video_fps} fps")
    else:
        processing_fps = fps
        print(f"Using specified frame rate: {fps} fps")
    
    # Calculate frame interval for extraction
    frame_interval = int(video_fps / processing_fps)
    if frame_interval < 1:
        frame_interval = 1
    
    # Create output videos
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = min(processing_fps, video_fps)
    
    main_video_path = os.path.join(output_dir, "annotated_video.mp4")
    segmentation_video_path = os.path.join(output_dir, "segmentation_with_predictions.mp4")
    
    main_video = cv2.VideoWriter(main_video_path, fourcc, output_fps, (width, height))
    segmentation_video = cv2.VideoWriter(segmentation_video_path, fourcc, output_fps, (width, height))
    
    # Create trajectory data file
    trajectory_file = os.path.join(output_dir, "trajectory_with_predictions.csv")
    with open(trajectory_file, 'w') as f:
        f.write("frame,timestamp,actual_x,actual_y,actual_z,predicted_x,predicted_y,predicted_z,error_3d\n")
    
    # For storing tracking data
    position_history = []
    position_data = []
    results = []
    
    # For processing
    frame_count = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Process only frames at the specified interval
            if frame_count % frame_interval == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
                
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get camera parameters
                camera_params = calculate_camera_parameters((height, width), camera_model, focal_length_mm)
                fx, fy = camera_params["fx"], camera_params["fy"]
                cx, cy = camera_params["cx"], camera_params["cy"]
                
                # Object detection
                seg_tensor = transforms.ToTensor()(frame_rgb).to(device)
                
                with torch.no_grad():
                    prediction = seg_model([seg_tensor])
                
                # Process results
                scores = prediction[0]['scores']
                
                if len(scores) > 0:
                    # Filter high confidence detections
                    high_conf_indices = torch.where(scores > 0.5)[0]
                    
                    if len(high_conf_indices) > 0:
                        # Look for airplane class (index 4)
                        airplane_indices = [i for i in high_conf_indices if prediction[0]['labels'][i].item() == 4]
                        
                        if airplane_indices:
                            best_idx = airplane_indices[0].item()
                        else:
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
                            from cv2 import minAreaRect, boxPoints
                            points = np.column_stack((x_indices, y_indices))
                            rect = minAreaRect(points)
                            box = boxPoints(rect)
                            box = box.astype(np.int32)
                            
                            rect_width, rect_height = rect[1]
                            
                            # Ensure width is the longer dimension
                            if rect_width < rect_height:
                                rect_width, rect_height = rect_height, rect_width
                            
                            # Depth estimation
                            pil_image = Image.fromarray(frame_rgb)
                            depth_transform = transforms.Compose([
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
                            depth_input = depth_transform(pil_image).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                depth_pred = depth_model(depth_input)
                            
                            depth_map = depth_pred.squeeze().cpu().numpy()
                            
                            # Normalize for visualization
                            depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                            depth_vis = (depth_vis * 255).astype(np.uint8)
                            depth_vis = cv2.resize(depth_vis, (width, height), interpolation=cv2.INTER_CUBIC)
                            
                            # Resize for calculation
                            depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_CUBIC)
                            aircraft_relative_depth = depth_map_resized[y_indices, x_indices].mean()
                            
                            # Distance calculation using reference dimensions
                            if known_jet_type and known_jet_type in FIGHTER_JET_DIMENSIONS:
                                jet_specs = FIGHTER_JET_DIMENSIONS[known_jet_type]
                                reference_length = jet_specs["length"]
                                aircraft_depth_meters = (reference_length * fx) / rect_width
                            else:
                                # Fallback to average jet length
                                avg_fighter_length = 17.0
                                aircraft_depth_meters = (avg_fighter_length * fx) / rect_width
                            
                            # Convert 2D centroid to 3D coordinates
                            x_meters = (u_centroid - cx) * aircraft_depth_meters / fx
                            y_meters = (v_centroid - cy) * aircraft_depth_meters / fy
                            z_meters = aircraft_depth_meters
                            
                            # Current actual position
                            actual_position = (x_meters, y_meters, z_meters)
                            
                            # Add to position history for LSTM
                            position_history.append(actual_position)
                            
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
                            
                            # If we have enough position history, make prediction with the LSTM model
                            predicted_position = None
                            error_3d = None
                            
                            if len(position_history) >= sequence_length:
                                # Get the most recent positions
                                recent_positions = position_history[-sequence_length:]
                                
                                try:
                                    # Try to make a prediction with all error handling
                                    try:
                                        # Scale data for LSTM
                                        scaled_positions = scaler.transform(recent_positions)
                                        
                                        # Convert to tensor
                                        X = torch.FloatTensor(scaled_positions).unsqueeze(0).to(device)
                                        
                                        # Make prediction
                                        with torch.no_grad():
                                            scaled_prediction = model(X).cpu().numpy()
                                        
                                        # Convert back to original scale
                                        predicted_position = scaler.inverse_transform(scaled_prediction)[0]
                                    except Exception as e:
                                        print(f"Error in LSTM prediction: {e}")
                                        # Fall back to simple prediction
                                        print("Using simple trajectory prediction instead")
                                        
                                        # Simple velocity-based prediction
                                        p1 = np.array(position_history[-2])
                                        p2 = np.array(position_history[-1])
                                        velocity = p2 - p1
                                        predicted_position = p2 + velocity
                                    
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
                                              (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                except Exception as e:
                                    print(f"Error handling prediction: {e}")
                            else:
                                # Not enough history for prediction yet
                                cv2.putText(segmentation,
                                          f"Collecting data: {len(position_history)}/{sequence_length}",
                                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                            
                            # Save to trajectory file
                            with open(trajectory_file, 'a') as f:
                                if predicted_position is not None:
                                    f.write(f"{processed_count},{frame_count/video_fps:.3f}," +
                                           f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f}," +
                                           f"{pred_x:.3f},{pred_y:.3f},{pred_z:.3f},{error_3d:.3f}\n")
                                else:
                                    f.write(f"{processed_count},{frame_count/video_fps:.3f}," +
                                           f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f},,,,\n")
                            
                            # Store data for results
                            position_row = {
                                "frame": processed_count,
                                "timestamp": frame_count/video_fps,
                                "actual_x": x_meters,
                                "actual_y": y_meters,
                                "actual_z": z_meters
                            }
                            
                            if predicted_position is not None:
                                position_row.update({
                                    "predicted_x": pred_x,
                                    "predicted_y": pred_y,
                                    "predicted_z": pred_z,
                                    "error_3d": error_3d
                                })
                            
                            position_data.append(position_row)
                            
                            # Convert RGB to BGR for OpenCV
                            segmentation_bgr = cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR)
                            
                            # Write to videos
                            main_video.write(frame)  # Original frame
                            segmentation_video.write(segmentation_bgr)  # Segmentation with predictions
                            
                            # Save frame for debugging if needed
                            if processed_count % 10 == 0:  # Save every 10th processed frame
                                frame_path = os.path.join(output_dir, f"frame_{processed_count:04d}.jpg")
                                cv2.imwrite(frame_path, segmentation_bgr)
                
                processed_count += 1
            
            frame_count += 1
    
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        cap.release()
        main_video.release()
        segmentation_video.release()
    
    print(f"Video processing complete!")
    print(f"Processed {processed_count} frames from {total_frames} total frames")
    
    # Create dataframe of results
    results_df = pd.DataFrame(position_data)
    results_path = os.path.join(output_dir, "position_prediction_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Create visualization of actual vs predicted trajectories if we have predictions
    if 'predicted_x' in results_df.columns:
        create_trajectory_comparison(results_df, output_dir)
    
    # Create results summary
    summary_path = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FIGHTER JET LSTM PREDICTION SUMMARY\n")
        f.write("==================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames processed: {processed_count}\n")
        f.write(f"LSTM model: {model_path}\n\n")
        
        if 'error_3d' in results_df.columns:
            avg_error = results_df['error_3d'].mean()
            max_error = results_df['error_3d'].max()
            min_error = results_df['error_3d'].min()
            
            f.write("PREDICTION STATISTICS\n")
            f.write(f"Average error: {avg_error:.3f} meters\n")
            f.write(f"Maximum error: {max_error:.3f} meters\n")
            f.write(f"Minimum error: {min_error:.3f} meters\n")
    
    # Return paths to output files
    return {
        "trajectory_csv": trajectory_file,
        "results_csv": results_path,
        "segmentation_video": segmentation_video_path,
        "main_video": main_video_path,
        "summary": summary_path
    }

# Main execution
if __name__ == "__main__":
    # First, upload the trained model
    print("Please upload your trained LSTM model (*.pt file):")
    uploaded_model = files.upload()
    model_path = list(uploaded_model.keys())[0]

    # Then upload the test video
    print("\nPlease upload your test video:")
    uploaded_video = files.upload()
    video_path = list(uploaded_video.keys())[0]

    # Set output directory
    output_dir = "/content/prediction_results"
    !mkdir -p {output_dir}

    # Run inference
    print("\nStarting inference with pre-trained model...")
    results = process_video_with_lstm_predictions(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir,
        camera_model="Sony Alpha 7R IV",
        focal_length_mm=200,
        known_jet_type="F-16 Fighting Falcon",
        fps=60,
        use_original_fps=True
    )

    # Compress results for download
    !zip -r /content/prediction_results.zip {output_dir}
    files.download('/content/prediction_results.zip')

    # Display the segmentation video with predictions
    from IPython.display import HTML
    from base64 import b64encode

    def display_video(video_path):
        video_file = open(video_path, "rb")
        video_bytes = video_file.read()
        video_b64 = b64encode(video_bytes).decode()
        video_html = f"""
        <video width="640" height="480" controls>
          <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        """
        return HTML(video_html)

    print("\nDisplaying segmentation video with predictions:")
    display_video(results["segmentation_video"])
