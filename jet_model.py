import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
        hidden_size = checkpoint.get('model_info', {}).get('hidden_size', 64)
        num_layers = checkpoint.get('model_info', {}).get('num_layers', 2)

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
        scaler = checkpoint.get('scaler', MinMaxScaler(feature_range=(-1, 1)))
        sequence_length = checkpoint.get('sequence_length', 10)

        print(f"Model loaded successfully from {model_path}")
        print(f"This model was trained on: {checkpoint.get('model_info', {}).get('date_trained', 'unknown date')}")

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
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
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