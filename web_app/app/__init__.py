from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv
from .config import (
    UPLOAD_FOLDER,
    RESULTS_FOLDER,
    MAX_CONTENT_LENGTH
)

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Import and register blueprint
from .routes import bp
app.register_blueprint(bp)

# Import routes after app creation to avoid circular imports
from . import routes
