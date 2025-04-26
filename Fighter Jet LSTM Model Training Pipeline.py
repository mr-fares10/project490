# Training Pipeline for Fighter Jet LSTM Model Using Multiple Videos

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
from datetime import datetime
import time
from google.colab import files
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import json
from tqdm import tqdm

# LSTM Model Definition
class JetPositionLSTM(nn.Module):
    """LSTM model for predicting the next position of a fighter jet"""
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout_rate=0.2):
        super(JetPositionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate
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

# Constants for fighter jet reference dimensions and camera parameters
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

# Aviation camera specifications
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

# Extract trajectory data from video
def extract_trajectory_from_video(video_path, output_dir, camera_model="Sony Alpha 7R IV",
                                focal_length_mm=200, known_jet_type="F-16 Fighting Falcon",
                                fps=30, use_original_fps=True):
    """
    Extract fighter jet trajectory data from a video file

    Returns:
        Path to the generated trajectory CSV file
    """
    print(f"Processing video: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine output file name
    video_basename = os.path.basename(video_path).split('.')[0]
    trajectory_file = os.path.join(output_dir, f"{video_basename}_trajectory.csv")

    # Load detection models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading detection models...")
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    seg_model.to(device)
    seg_model.eval()

    depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    depth_model.to(device)
    depth_model.eval()
    print("Detection models loaded")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video info: {width}x{height}, {video_fps} fps, {total_frames} frames")

    # Determine processing frame rate
    if use_original_fps:
        processing_fps = video_fps
    else:
        processing_fps = fps

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / processing_fps))

    # Create trajectory data file
    with open(trajectory_file, 'w') as f:
        f.write("frame,timestamp,x,y,z\n")

    # For tracking
    position_data = []
    frame_count = 0
    processed_count = 0

    # Process video frames
    progress_bar = tqdm(total=total_frames, desc="Extracting trajectory")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Update progress
            progress_bar.update(1)

            # Process only frames at the specified interval
            if frame_count % frame_interval == 0:
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

                        # Get mask
                        mask = prediction[0]['masks'][best_idx, 0].cpu().numpy()

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
                            rect_width, rect_height = rect[1]

                            # Ensure width is the longer dimension
                            if rect_width < rect_height:
                                rect_width, rect_height = rect_height, rect_width

                            # Depth estimation with MiDaS
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

                            # Resize for calculation
                            depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_CUBIC)

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

                            # Save to trajectory file
                            with open(trajectory_file, 'a') as f:
                                f.write(f"{processed_count},{frame_count/video_fps:.3f}," +
                                       f"{x_meters:.3f},{y_meters:.3f},{z_meters:.3f}\n")

                            # Store data
                            position_row = {
                                "frame": processed_count,
                                "timestamp": frame_count/video_fps,
                                "x": x_meters,
                                "y": y_meters,
                                "z": z_meters
                            }
                            position_data.append(position_row)

                processed_count += 1

            frame_count += 1

    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close resources
        cap.release()
        progress_bar.close()

    print(f"Extracted {processed_count} positions from {video_path}")

    # Create summary file
    summary_path = os.path.join(output_dir, f"{video_basename}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Processed frames: {processed_count}\n")
        f.write(f"Positions extracted: {len(position_data)}\n")
        f.write(f"Camera model: {camera_model}\n")
        f.write(f"Focal length: {focal_length_mm}mm\n")
        f.write(f"Aircraft type: {known_jet_type}\n")

    return trajectory_file, len(position_data)

# Prepare data for training from multiple trajectory files
def prepare_training_data(trajectory_files, sequence_length):
    """
    Prepare training data from multiple trajectory files

    Args:
        trajectory_files: List of paths to trajectory CSV files
        sequence_length: Number of timesteps to use for input sequences

    Returns:
        X_train, y_train, X_val, y_val, scaler
    """
    all_data = []

    # Process each trajectory file
    print(f"Preparing training data from {len(trajectory_files)} trajectory files...")

    for file_path in trajectory_files:
        try:
            # Load trajectory data
            df = pd.read_csv(file_path)

            # Check for required columns
            if not all(col in df.columns for col in ['x', 'y', 'z']):
                print(f"Warning: {file_path} missing coordinate columns, checking alternative names...")
                # Try alternative column names
                coord_cols = []
                if 'actual_x' in df.columns:
                    df['x'] = df['actual_x']
                    coord_cols.append('actual_x')
                if 'actual_y' in df.columns:
                    df['y'] = df['actual_y']
                    coord_cols.append('actual_y')
                if 'actual_z' in df.columns:
                    df['z'] = df['actual_z']
                    coord_cols.append('actual_z')

                if len(coord_cols) < 3:
                    print(f"Skipping {file_path}: couldn't find coordinate columns")
                    continue

            # Get position data
            positions = df[['x', 'y', 'z']].values

            # Skip if too few positions
            if len(positions) < sequence_length + 1:
                print(f"Skipping {file_path}: too few positions ({len(positions)})")
                continue

            # Add to all data
            all_data.append(positions)
            print(f"Added {len(positions)} positions from {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_data:
        raise ValueError("No usable trajectory data found in the provided files")

    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Create sequences and labels
    X_sequences = []
    y_labels = []

    # Process each trajectory separately to avoid creating sequences across different videos
    for trajectory in all_data:
        # Scale the trajectory
        scaled_trajectory = scaler.fit_transform(trajectory)

        # Create sequences
        for i in range(len(scaled_trajectory) - sequence_length):
            X_sequences.append(scaled_trajectory[i:i+sequence_length])
            y_labels.append(scaled_trajectory[i+sequence_length])

    # Convert to numpy arrays
    X = np.array(X_sequences)
    y = np.array(y_labels)

    print(f"Created {len(X)} sequences for training/validation")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val, scaler

# Train LSTM model from scratch or continue training
def train_lstm_model(X_train, y_train, X_val, y_val, sequence_length,
                    hidden_size=64, num_layers=2, batch_size=32,
                    learning_rate=0.001, epochs=100, patience=20,
                    existing_model_path=None, output_dir=None):
    """
    Train an LSTM model for fighter jet position prediction

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sequence_length: Input sequence length
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        batch_size: Mini-batch size
        learning_rate: Learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
        existing_model_path: Path to existing model for continued training
        output_dir: Directory to save checkpoints

    Returns:
        Trained model, scaler, and training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create model
    if existing_model_path and os.path.exists(existing_model_path):
        print(f"Loading existing model from {existing_model_path}")
        try:
            # Load checkpoint
            checkpoint = torch.load(existing_model_path, map_location=device)

            # Get model info if available
            try:
                model_info = checkpoint['model_info']
                hidden_size = model_info.get('hidden_size', hidden_size)
                num_layers = model_info.get('num_layers', num_layers)
                print(f"Using model architecture: hidden_size={hidden_size}, num_layers={num_layers}")
            except:
                print("Using default model architecture")

            # Create model
            model = JetPositionLSTM(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=3
            ).to(device)

            # Load state dict
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model state loaded successfully")
            except:
                try:
                    model.load_state_dict(checkpoint)
                    print("Loaded state dict directly")
                except:
                    print("Could not load model state, using newly initialized model")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model instead")
            model = JetPositionLSTM(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=3
            ).to(device)
    else:
        print("Training new model")
        model = JetPositionLSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3
        ).to(device)

    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    # Training loop
    start_time = time.time()
    print(f"Starting training for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Calculate average loss
        train_loss = train_loss / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"LR: {current_lr:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0

            # Save checkpoint
            if output_dir:
                checkpoint_path = os.path.join(output_dir, f"lstm_checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_info': {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")

        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state")

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    return model, history

# Save trained model
def save_trained_model(model, scaler, sequence_length, history, output_dir):
    """Save the trained model with all components"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create model info
    model_info = {
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sequence_length': sequence_length
    }

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': model_info,
        'scaler': scaler,
        'sequence_length': sequence_length,
        'training_history': history
    }

    # Save model
    model_path = os.path.join(output_dir, f"jet_lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    # Learning rate curve
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save training summary
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("FIGHTER JET LSTM TRAINING SUMMARY\n")
        f.write("================================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model architecture: {model.hidden_size} hidden units, {model.num_layers} layers\n")
        f.write(f"Sequence length: {sequence_length}\n\n")

        f.write("TRAINING RESULTS\n")
        f.write(f"Final training loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"Best validation loss: {min(history['val_loss']):.6f}\n")
        f.write(f"Total epochs: {len(history['train_loss'])}\n")

    return model_path

# Evaluate model on a test video
def evaluate_model(model_path, test_video_path, output_dir):
    """Evaluate the trained model on a test video"""

    # Process test video to extract trajectory
    trajectory_file, num_positions = extract_trajectory_from_video(
        test_video_path, output_dir
    )

    if num_positions == 0:
        print("No positions extracted from test video")
        return None

    # Process video with predictions
    from fighter_jet_tracking_code import process_video_with_lstm_predictions

    results = process_video_with_lstm_predictions(
        video_path=test_video_path,
        model_path=model_path,
        output_dir=output_dir,
        camera_model="Sony Alpha 7R IV",
        focal_length_mm=200,
        known_jet_type="F-16 Fighting Falcon",
        fps=60,
        use_original_fps=True
    )

    return results

# Main function to run the multi-video training pipeline
def main_multi_video_training():
    print("=== Fighter Jet LSTM Training Pipeline with Multiple Videos ===\n")

    # Create output directory
    output_dir = "/content/fighter_jet_lstm_training"
    os.makedirs(output_dir, exist_ok=True)

    # Ask if the user wants to upload videos or trajectory CSVs
    use_videos = input("Do you want to upload videos directly (y) or trajectory CSV files (n)? ").lower() == 'y'

    trajectory_files = []

    if use_videos:
        # Upload training videos
        print("\nPlease upload training videos (you can select multiple files):")
        uploaded_videos = files.upload()
        video_paths = list(uploaded_videos.keys())

        if not video_paths:
            print("No videos uploaded")
            return

        print(f"Processing {len(video_paths)} videos...")

        # Extract trajectories from each video
        for video_path in video_paths:
            trajectory_file, num_positions = extract_trajectory_from_video(
                video_path, output_dir
            )

            if num_positions > 0:
                trajectory_files.append(trajectory_file)
    else:
        # Upload trajectory CSV files
        print("\nPlease upload trajectory CSV files (you can select multiple files):")
        uploaded_csvs = files.upload()
        trajectory_files = list(uploaded_csvs.keys())

        if not trajectory_files:
            print("No CSV files uploaded")
            return

    # Set training parameters
    sequence_length = int(input("\nEnter sequence length (default: 10): ") or 10)
    hidden_size = int(input("Enter hidden size (default: 64): ") or 64)
    num_layers = int(input("Enter number of LSTM layers (default: 2): ") or 2)
    batch_size = int(input("Enter batch size (default: 32): ") or 32)
    learning_rate = float(input("Enter learning rate (default: 0.001): ") or 0.001)
    epochs = int(input("Enter maximum epochs (default: 100): ") or 100)

    # Ask for existing model
    use_existing = input("\nDo you want to continue training from an existing model? (y/n): ").lower() == 'y'
    existing_model_path = None

    if use_existing:
        print("Please upload the existing model (.pt file):")
        uploaded_model = files.upload()
        if uploaded_model:
            existing_model_path = list(uploaded_model.keys())[0]

    # Prepare training data
    X_train, y_train, X_val, y_val, scaler = prepare_training_data(trajectory_files, sequence_length)

    # Train model
    model, history = train_lstm_model(
        X_train, y_train, X_val, y_val, sequence_length,
        hidden_size=hidden_size, num_layers=num_layers,
        batch_size=batch_size, learning_rate=learning_rate,
        epochs=epochs, existing_model_path=existing_model_path,
        output_dir=output_dir
    )

    # Save trained model
    model_path = save_trained_model(model, scaler, sequence_length, history, output_dir)

    # Ask for evaluation
    run_eval = input("\nDo you want to evaluate the model on a test video? (y/n): ").lower() == 'y'

    if run_eval:
        print("Please upload a test video:")
        uploaded_test = files.upload()
        if uploaded_test:
            test_video_path = list(uploaded_test.keys())[0]
            results = evaluate_model(model_path, test_video_path, output_dir)

    # Compress results for download
    print("\nCompressing results for download...")
    !zip -r /content/fighter_jet_lstm_results.zip {output_dir}
    files.download('/content/fighter_jet_lstm_results.zip')

    print("\nTraining pipeline completed!")
    print(f"Trained model saved to: {model_path}")
    print("Download the ZIP file for complete results")

# Function to mount Google Drive and get file paths
def setup_google_drive():
    """Mount Google Drive and get file paths from it"""
    from google.colab import drive

    # Mount Google Drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")

    # Ask for folder path
    folder_path = input("Enter the folder path in Google Drive where videos are stored (e.g., MyDrive/fighter_jet_videos): ")
    full_path = f"/content/drive/{folder_path}"

    if not os.path.exists(full_path):
        print(f"Error: Path {full_path} not found")
        return []

    # List video files in the folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for file in os.listdir(full_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(full_path, file))

    if not video_files:
        print(f"No video files found in {full_path}")
        return []

    print(f"Found {len(video_files)} video files:")
    for i, video_path in enumerate(video_files):
        print(f"{i+1}. {os.path.basename(video_path)}")

    # Ask which videos to use
    selection = input("Enter video numbers to use (comma-separated, or 'all' for all videos): ")

    if selection.lower() == 'all':
        return video_files
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_videos = [video_files[idx] for idx in indices if 0 <= idx < len(video_files)]
            return selected_videos
        except:
            print("Invalid selection, using all videos")
            return video_files

# Enhanced main function with Google Drive integration
def enhanced_multi_video_training():
    print("=== Fighter Jet LSTM Training Pipeline with Multiple Videos ===\n")

    # Create output directory
    output_dir = "/content/fighter_jet_lstm_training"
    os.makedirs(output_dir, exist_ok=True)

    # Ask about source
    source_option = input("Select data source:\n1. Upload videos/CSVs directly\n2. Use files from Google Drive\nEnter option (1 or 2): ")

    trajectory_files = []

    if source_option == '2':
        # Use Google Drive
        video_paths = setup_google_drive()

        if not video_paths:
            print("No videos selected from Google Drive")
            direct_upload = input("Do you want to try direct upload instead? (y/n): ").lower() == 'y'
            if not direct_upload:
                return
            source_option = '1'
        else:
            use_videos = True

    if source_option == '1':
        # Ask if the user wants to upload videos or trajectory CSVs
        use_videos = input("Do you want to upload videos directly (y) or trajectory CSV files (n)? ").lower() == 'y'

        if use_videos:
            # Upload training videos
            print("\nPlease upload training videos (you can select multiple files):")
            uploaded_videos = files.upload()
            video_paths = list(uploaded_videos.keys())

            if not video_paths:
                print("No videos uploaded")
                return
        else:
            # Upload trajectory CSV files
            print("\nPlease upload trajectory CSV files (you can select multiple files):")
            uploaded_csvs = files.upload()
            trajectory_files = list(uploaded_csvs.keys())

            if not trajectory_files:
                print("No CSV files uploaded")
                return

    # Process videos if needed
    if use_videos:
        print(f"Processing {len(video_paths)} videos...")

        # Extract trajectories from each video
        for video_path in video_paths:
            trajectory_file, num_positions = extract_trajectory_from_video(
                video_path, output_dir
            )

            if num_positions > 0:
                trajectory_files.append(trajectory_file)

    if not trajectory_files:
        print("No usable trajectory data found")
        return

    # Set training parameters
    sequence_length = int(input("\nEnter sequence length (default: 10): ") or 10)
    hidden_size = int(input("Enter hidden size (default: 64): ") or 64)
    num_layers = int(input("Enter number of LSTM layers (default: 2): ") or 2)
    batch_size = int(input("Enter batch size (default: 32): ") or 32)
    learning_rate = float(input("Enter learning rate (default: 0.001): ") or 0.001)
    epochs = int(input("Enter maximum epochs (default: 100): ") or 100)

    # Ask for existing model
    use_existing = input("\nDo you want to continue training from an existing model? (y/n): ").lower() == 'y'
    existing_model_path = None

    if use_existing:
        existing_source = input("Upload model file (u) or select from Google Drive (g)? ").lower()

        if existing_source == 'g':
            drive_model_path = input("Enter the full path to the model in Google Drive: ")
            if os.path.exists(drive_model_path):
                existing_model_path = drive_model_path
            else:
                print(f"Model not found at {drive_model_path}")
        else:
            print("Please upload the existing model (.pt file):")
            uploaded_model = files.upload()
            if uploaded_model:
                existing_model_path = list(uploaded_model.keys())[0]

    # Prepare training data
    X_train, y_train, X_val, y_val, scaler = prepare_training_data(trajectory_files, sequence_length)

    # Train model
    model, history = train_lstm_model(
        X_train, y_train, X_val, y_val, sequence_length,
        hidden_size=hidden_size, num_layers=num_layers,
        batch_size=batch_size, learning_rate=learning_rate,
        epochs=epochs, existing_model_path=existing_model_path,
        output_dir=output_dir
    )

    # Save trained model
    model_path = save_trained_model(model, scaler, sequence_length, history, output_dir)

    # Save to Google Drive if needed
    if source_option == '2':
        drive_save_path = input("\nEnter Google Drive path to save the model (or press Enter to skip): ")
        if drive_save_path:
            full_drive_path = f"/content/drive/{drive_save_path}"
            os.makedirs(os.path.dirname(full_drive_path), exist_ok=True)
            shutil.copy(model_path, full_drive_path)
            print(f"Model also saved to Google Drive at: {full_drive_path}")

    # Ask for evaluation
    run_eval = input("\nDo you want to evaluate the model on a test video? (y/n): ").lower() == 'y'

    if run_eval:
        if source_option == '2':
            eval_source = input("Upload test video (u) or select from Google Drive (g)? ").lower()

            if eval_source == 'g':
                test_video_options = setup_google_drive()
                if test_video_options:
                    test_video_idx = int(input(f"Enter video number (1-{len(test_video_options)}): ")) - 1
                    if 0 <= test_video_idx < len(test_video_options):
                        test_video_path = test_video_options[test_video_idx]
                        results = evaluate_model(model_path, test_video_path, output_dir)
            else:
                print("Please upload a test video:")
                uploaded_test = files.upload()
                if uploaded_test:
                    test_video_path = list(uploaded_test.keys())[0]
                    results = evaluate_model(model_path, test_video_path, output_dir)
        else:
            print("Please upload a test video:")
            uploaded_test = files.upload()
            if uploaded_test:
                test_video_path = list(uploaded_test.keys())[0]
                results = evaluate_model(model_path, test_video_path, output_dir)

    # Compress results for download
    print("\nCompressing results for download...")
    !zip -r /content/fighter_jet_lstm_results.zip {output_dir}
    files.download('/content/fighter_jet_lstm_results.zip')

    print("\nTraining pipeline completed!")
    print(f"Trained model saved to: {model_path}")
    print("Download the ZIP file for complete results")

# Run the enhanced training pipeline
if __name__ == "__main__":
    # Import missing modules
    import shutil

    # First, import the tracking code module to enable evaluation
    try:
        import fighter_jet_tracking_code
    except ImportError:
        print("Cannot find fighter_jet_tracking_code module. Creating it...")

        # Write tracking code to file
        tracking_code_path = "/content/fighter_jet_tracking_code.py"
        with open(tracking_code_path, 'w') as f:
            f.write("""
# This is a stub for the fighter jet tracking code
# It will be imported by the training pipeline for evaluation
# See the full implementation in the main code artifact

def process_video_with_lstm_predictions(video_path, model_path, output_dir="/content/output",
                                      camera_model="Sony Alpha 7R IV", focal_length_mm=200,
                                      known_jet_type="F-16 Fighting Falcon", fps=60, use_original_fps=True):
    print(f"Processing video {video_path} with model {model_path}")
    print("This is just a stub function - for full implementation, use the main code")

    # Create dummy output paths
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return {
        "trajectory_csv": os.path.join(output_dir, "dummy_trajectory.csv"),
        "results_csv": os.path.join(output_dir, "dummy_results.csv"),
        "segmentation_video": os.path.join(output_dir, "dummy_segmentation.mp4"),
        "main_video": os.path.join(output_dir, "dummy_main.mp4"),
        "summary": os.path.join(output_dir, "dummy_summary.txt")
    }
""")

    # Run the training pipeline
    enhanced_multi_video_training()
