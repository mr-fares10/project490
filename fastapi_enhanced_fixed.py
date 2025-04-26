from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import logging
from typing import Optional
from pathlib import Path
import sys
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our advanced processor
from advanced_processor import AdvancedVideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fighter-jet-app")

# Define base directories using absolute paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, "web_app", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "web_app", "results")
MODEL_FOLDER = os.path.join(BASE_DIR, "web_app", "app", "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "web_app", "app", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "app", "static")

# Create a separate directory for example videos to avoid conflicts
EXAMPLES_FOLDER = os.path.join(BASE_DIR, "examples")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(EXAMPLES_FOLDER, exist_ok=True)

# Define constants
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
ALLOWED_MODEL_EXTENSIONS = {'pt', 'pth'}

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Fighter jet dimensions
FIGHTER_JET_DIMENSIONS = {
    'F-16': {'length': 15.03, 'wingspan': 9.96, 'height': 4.88},
    'F-18': {'length': 17.1, 'wingspan': 13.7, 'height': 4.7},
    'F-22': {'length': 18.92, 'wingspan': 13.56, 'height': 5.08},
    'F-35': {'length': 15.7, 'wingspan': 10.7, 'height': 4.36}
}

# Aviation camera specifications
AVIATION_CAMERAS = {
    'Standard': {'sensor_width': 36.0, 'sensor_height': 24.0},
    'High-Speed': {'sensor_width': 35.9, 'sensor_height': 23.9},
    'Tracking': {'sensor_width': 32.0, 'sensor_height': 24.0}
}

# Initialize FastAPI app
app = FastAPI(
    title="Fighter Jet Position Prediction",
    description="Web service for processing videos to predict fighter jet positions"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Mounted static files from: {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found at: {STATIC_DIR}")

# Mount examples folder as a separate static files location
if os.path.exists(EXAMPLES_FOLDER):
    app.mount("/examples", StaticFiles(directory=EXAMPLES_FOLDER), name="examples")
    logger.info(f"Mounted examples from: {EXAMPLES_FOLDER}")
else:
    logger.warning(f"Examples directory not found at: {EXAMPLES_FOLDER}")

# Initialize templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Simple job manager
class SimpleJobManager:
    def __init__(self, results_folder):
        self.results_folder = results_folder
        self.jobs = {}
        
    def create_job(self, job_id, video_path, model_path):
        job_data = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'video_path': video_path,
            'model_path': model_path
        }
        self.jobs[job_id] = job_data
        return job_data
        
    def update_job_progress(self, job_id, progress):
        if job_id in self.jobs:
            self.jobs[job_id]['progress'] = progress
            
    def complete_job(self, job_id, results):
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['progress'] = 100
            self.jobs[job_id]['results'] = results
            
    def fail_job(self, job_id, error_message):
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = error_message
            
    def get_job_status(self, job_id):
        return self.jobs.get(job_id)
        
    def cancel_job(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'cancelled'
            return True
        return False

# Initialize job manager
job_manager = SimpleJobManager(RESULTS_FOLDER)

# Check if a file has an allowed extension
def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Save the enhanced template if it doesn't exist
def ensure_template_exists():
    # Template path for the enhanced template
    template_path = os.path.join(TEMPLATES_DIR, "enhanced.html")
    
    if not os.path.exists(template_path):
        # If the template doesn't exist, read it from our artifact
        with open(template_path, "w") as f:
            # Add your enhanced HTML content here (it's very large, so I'm not including the full content)
            f.write("<!-- Enhanced template will be saved here -->")
            # Note: You should manually save the enhanced.html to this location

    # Create example videos in the separate examples folder
    example_files = {
        "f16_tracking.mp4": "F-16 Tracking Example",
        "f22_tracking.mp4": "F-22 Raptor Example",
        "f35_tracking.mp4": "F-35 Lightning II Example",
        "example1_poster.jpg": "F-16 Poster",
        "example2_poster.jpg": "F-22 Poster",
        "example3_poster.jpg": "F-35 Poster"
    }
    
    for file_name, description in example_files.items():
        file_path = os.path.join(EXAMPLES_FOLDER, file_name)
        if not os.path.exists(file_path):
            # Create empty placeholder file
            try:
                Path(file_path).touch()
                logger.info(f"Created placeholder for example file: {file_path}")
            except Exception as e:
                logger.error(f"Could not create placeholder file: {e}")

# Make sure template exists when app starts
ensure_template_exists()

# Update the HTML template to use the new examples path
def update_template_paths():
    template_path = os.path.join(TEMPLATES_DIR, "enhanced.html")
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                content = f.read()
                
            # Replace static example paths with the new mount point
            updated_content = content.replace('/static/examples/', '/examples/')
            updated_content = updated_content.replace('/static/example1_poster.jpg', '/examples/example1_poster.jpg')
            updated_content = updated_content.replace('/static/example2_poster.jpg', '/examples/example2_poster.jpg')
            updated_content = updated_content.replace('/static/example3_poster.jpg', '/examples/example3_poster.jpg')
            
            with open(template_path, 'w') as f:
                f.write(updated_content)
                
            logger.info("Updated template paths to use separate examples folder")
        except Exception as e:
            logger.error(f"Error updating template paths: {e}")

# Update template paths on startup
update_template_paths()

# Routes
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Render the main web interface."""
    return templates.TemplateResponse("enhanced.html", {"request": request})

@app.get("/simple", response_class=HTMLResponse)
async def simple_interface(request: Request):
    """Render the simple web interface."""
    return templates.TemplateResponse("simple.html", {"request": request})

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model: Optional[UploadFile] = None,
    jet_type: str = Form("F-16"),
    camera_model: str = Form("Standard"),
    focal_length: float = Form(200.0)
):
    """Upload a video file and optional model file for processing."""
    try:
        # Check file extension
        if not allowed_file(video.filename, ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(
                status_code=400, 
                detail=f"Video file type not allowed. Allowed types: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )
        
        # Create job ID and directories
        job_id = str(uuid.uuid4())
        job_upload_dir = os.path.join(UPLOAD_FOLDER, job_id)
        job_results_dir = os.path.join(RESULTS_FOLDER, job_id)
        
        os.makedirs(job_upload_dir, exist_ok=True)
        os.makedirs(job_results_dir, exist_ok=True)
        
        # Save video file
        video_path = os.path.join(job_upload_dir, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Save model file if provided
        model_path = None
        if model and model.filename:
            if not allowed_file(model.filename, ALLOWED_MODEL_EXTENSIONS):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model file type not allowed. Allowed types: {', '.join(ALLOWED_MODEL_EXTENSIONS)}"
                )
            
            model_path = os.path.join(job_upload_dir, model.filename)
            with open(model_path, "wb") as buffer:
                shutil.copyfileobj(model.file, buffer)
        else:
            # Use default model if available
            default_model = os.path.join(MODEL_FOLDER, "jet_lstm_model.pt")
            if os.path.exists(default_model):
                model_path = default_model
        
        # Create job
        job_data = job_manager.create_job(job_id, video_path, model_path)
        
        # Start processing in background
        background_tasks.add_task(
            process_video,
            job_id=job_id, 
            video_path=video_path,
            model_path=model_path,
            output_dir=job_results_dir,
            jet_type=jet_type,
            camera_model=camera_model,
            focal_length=focal_length
        )
        
        return JSONResponse({
            "status": "success",
            "message": "Processing started",
            "job_id": job_id
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_video(
    job_id: str, 
    video_path: str,
    model_path: str,
    output_dir: str,
    jet_type: str,
    camera_model: str,
    focal_length: float
):
    """Process video in background using our advanced processor."""
    try:
        # Initialize advanced processor
        processor = AdvancedVideoProcessor(
            video_path=video_path,
            output_dir=output_dir,
            model_path=model_path,
            jet_type=jet_type,
            camera_model=camera_model,
            focal_length=focal_length
        )
        
        # Define callbacks
        def progress_callback(progress: float):
            job_manager.update_job_progress(job_id, progress)
        
        def completion_callback():
            # Get statistics if available
            prediction_stats = None
            try:
                # Try to read processing summary
                summary_path = os.path.join(output_dir, 'processing_summary.txt')
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        content = f.read()
                        if "PREDICTION STATISTICS" in content:
                            # Extract error statistics
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if "PREDICTION STATISTICS" in line:
                                    avg_line = lines[i+1]
                                    max_line = lines[i+2]
                                    min_line = lines[i+3]
                                    
                                    avg_error = float(avg_line.split(': ')[1].split(' ')[0])
                                    max_error = float(max_line.split(': ')[1].split(' ')[0])
                                    min_error = float(min_line.split(': ')[1].split(' ')[0])
                                    
                                    prediction_stats = {
                                        'avg_error': avg_error,
                                        'max_error': max_error,
                                        'min_error': min_error
                                    }
                                    break
            except Exception as e:
                logger.error(f"Error reading prediction stats: {str(e)}")
            
            # Create results object
            results = {
                "video_info": {
                    "fps": processor.fps,
                    "total_frames": processor.total_frames,
                    "width": processor.width,
                    "height": processor.height,
                },
                "files": {
                    "main_video": "output_annotated.mp4",
                    "segmentation_video": "segmentation_with_predictions.mp4",
                    "trajectory_csv": "trajectory_with_predictions.csv",
                    "trajectory_3d": "trajectory_comparison_3d.png",
                    "trajectory_2d": "trajectory_comparison_2d.png",
                    "error_graph": "prediction_error.png",
                    "summary": "processing_summary.txt"
                }
            }
            
            # Add prediction stats if available
            if prediction_stats:
                results["prediction_stats"] = prediction_stats
            
            job_manager.complete_job(job_id, results)
        
        def error_callback(error_message: str):
            job_manager.fail_job(job_id, error_message)
        
        # Process video
        processor.process_async(
            progress_callback=progress_callback,
            completion_callback=completion_callback,
            error_callback=error_callback
        )
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        job_manager.fail_job(job_id, str(e))

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(job_status)

@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """Get the latest frame preview for a job."""
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check for processor instance
    if job_status.get('status') == 'processing':
        try:
            # Check for preview frames
            frames_dir = os.path.join(RESULTS_FOLDER, job_id, 'frames')
            if os.path.exists(frames_dir):
                frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')], 
                              key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                if frames:
                    latest_frame = frames[-1]
                    return JSONResponse({
                        "preview_available": True,
                        "preview_path": f"frames/{latest_frame}"
                    })
            
            return JSONResponse({"preview_available": False})
            
        except Exception as e:
            logger.error(f"Error getting preview: {str(e)}")
            return JSONResponse({"preview_available": False})
    
    return JSONResponse({"preview_available": False})

@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse({"status": "cancelled"})

@app.get("/api/results/{job_id}/{file_path:path}")
async def get_result_file(job_id: str, file_path: str):
    """Get a result file from a job."""
    full_path = os.path.join(RESULTS_FOLDER, job_id, file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(full_path)

@app.get("/api/jet-types")
async def get_jet_types():
    """Get available fighter jet types."""
    return JSONResponse({"jet_types": FIGHTER_JET_DIMENSIONS})

@app.get("/api/camera-models")
async def get_camera_models():
    """Get available camera models."""
    return JSONResponse({"camera_models": AVIATION_CAMERAS})
# Add these routes to your FastAPI application
# Make sure they are placed before the "if __name__ == "__main__"" section

@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """Render the debug page for API testing."""
    return templates.TemplateResponse("debug.html", {"request": request})

# Ensure the preview API is working correctly
@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """Get the latest frame preview for a job."""
    job_status = job_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check for preview frames
    frames_dir = os.path.join(RESULTS_FOLDER, job_id, 'frames')
    if os.path.exists(frames_dir):
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')], 
                      key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if frames:
            latest_frame = frames[-1]
            return JSONResponse({
                "preview_available": True,
                "preview_path": f"frames/{latest_frame}"
            })
    
    return JSONResponse({"preview_available": False})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_enhanced_fixed:app", host="127.0.0.1", port=8080, reload=True)
    