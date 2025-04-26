import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler
import torch.serialization
import numpy._core.multiarray as multiarray

# Add necessary safe globals
torch.serialization.add_safe_globals([MinMaxScaler, multiarray._reconstruct])

class VideoProcessor:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the trained model from the specified path.
        Implement the appropriate loading logic based on your model type (TensorFlow/PyTorch).
        """
        try:
            # Example for TensorFlow model
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for model input.
        Modify this method according to your model's input requirements.
        """
        # Example preprocessing
        frame = cv2.resize(frame, (224, 224))  # Resize to model input size
        frame = frame / 255.0  # Normalize
        return frame

    def process_video(self, video_path):
        """
        Process a video file and return predictions.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if self.model is None:
            raise ValueError("Model not loaded")

        predictions = {
            'positions': [],
            'timestamps': []
        }

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get timestamp
                timestamp = frame_count / fps
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Make prediction
                # Modify this part according to your model's prediction logic
                prediction = self.model.predict(np.expand_dims(processed_frame, axis=0))
                
                # Store results
                predictions['positions'].append(prediction[0].tolist())  # Assuming prediction is [x, y, z]
                predictions['timestamps'].append(f"{timestamp:.2f}")
                
                frame_count += 1

            cap.release()
            return predictions

        except Exception as e:
            raise RuntimeError(f"Error processing video: {e}")

    def cleanup(self):
        """
        Clean up resources
        """
        if hasattr(self, 'model'):
            del self.model
        tf.keras.backend.clear_session() 

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with 10 hidden units (to match the saved model)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # Match the saved model's size
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 3)  # hidden_size -> 3 output dimensions
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_model(model_path, device='cpu'):
    """Load the LSTM model from the given path."""
    model = LSTMPredictor()
    
    # Load with weights_only=False since we trust our own model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model 