import os

# Application root directory
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File upload settings
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
RESULTS_FOLDER = os.path.join(APP_ROOT, 'results')
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
ALLOWED_MODEL_EXTENSIONS = {'pt', 'pth'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Fighter jet dimensions (in meters)
FIGHTER_JET_DIMENSIONS = {
    'F-16': {
        'length': 15.03,
        'wingspan': 9.96,
        'height': 4.88
    },
    'F-18': {
        'length': 17.1,
        'wingspan': 13.7,
        'height': 4.7
    },
    'F-22': {
        'length': 18.92,
        'wingspan': 13.56,
        'height': 5.08
    },
    'F-35': {
        'length': 15.7,
        'wingspan': 10.7,
        'height': 4.36
    }
}

# Aviation camera specifications
AVIATION_CAMERAS = {
    'Standard': {
        'sensor_width': 36.0,  # mm
        'sensor_height': 24.0,  # mm
        'resolution_width': 1920,
        'resolution_height': 1080,
        'focal_length_range': (20, 800)  # mm
    },
    'High-Speed': {
        'sensor_width': 35.9,
        'sensor_height': 23.9,
        'resolution_width': 3840,
        'resolution_height': 2160,
        'focal_length_range': (25, 1000)
    },
    'Tracking': {
        'sensor_width': 32.0,
        'sensor_height': 24.0,
        'resolution_width': 1920,
        'resolution_height': 1080,
        'focal_length_range': (16, 600)
    }
}