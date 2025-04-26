import os
import json
import shutil
from threading import Lock
from datetime import datetime
from typing import Dict, Optional

class JobManager:
    def __init__(self, results_folder: str):
        self.results_folder = results_folder
        self.jobs: Dict[str, dict] = {}
        self.jobs_lock = Lock()

    def create_job(self, job_id: str, video_path: str, model_path: str) -> dict:
        """Create a new processing job."""
        job_data = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'video_path': video_path,
            'model_path': model_path,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'error': None,
            'results': {}
        }
        
        with self.jobs_lock:
            self.jobs[job_id] = job_data
            self._save_job_status(job_id, job_data)
        
        return job_data

    def update_job_progress(self, job_id: str, progress: float) -> None:
        """Update the progress of a job."""
        with self.jobs_lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                self._save_job_status(job_id, self.jobs[job_id])

    def complete_job(self, job_id: str, results: dict) -> None:
        """Mark a job as completed with results."""
        with self.jobs_lock:
            if job_id in self.jobs:
                self.jobs[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'end_time': datetime.now().isoformat(),
                    'results': results
                })
                self._save_job_status(job_id, self.jobs[job_id])

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with an error message."""
        with self.jobs_lock:
            if job_id in self.jobs:
                self.jobs[job_id].update({
                    'status': 'failed',
                    'end_time': datetime.now().isoformat(),
                    'error': error_message
                })
                self._save_job_status(job_id, self.jobs[job_id])

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get the current status of a job."""
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job and clean up its resources."""
        with self.jobs_lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            job['status'] = 'cancelled'
            job['end_time'] = datetime.now().isoformat()
            
            # Clean up job files
            job_dir = os.path.join(self.results_folder, job_id)
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)

            self._save_job_status(job_id, job)
            return True

    def _save_job_status(self, job_id: str, job_data: dict) -> None:
        """Save job status to a file."""
        job_dir = os.path.join(self.results_folder, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        status_file = os.path.join(job_dir, 'status.json')
        with open(status_file, 'w') as f:
            json.dump(job_data, f, indent=2)

    def load_job_statuses(self) -> None:
        """Load all job statuses from the results folder."""
        if not os.path.exists(self.results_folder):
            return

        for job_id in os.listdir(self.results_folder):
            status_file = os.path.join(self.results_folder, job_id, 'status.json')
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        job_data = json.load(f)
                        self.jobs[job_id] = job_data
                except (json.JSONDecodeError, IOError):
                    continue

    def cleanup_old_jobs(self, max_age_days: int = 7) -> None:
        """Clean up jobs older than max_age_days."""
        current_time = datetime.now()
        
        with self.jobs_lock:
            for job_id, job_data in list(self.jobs.items()):
                try:
                    start_time = datetime.fromisoformat(job_data['start_time'])
                    age = (current_time - start_time).days
                    
                    if age > max_age_days:
                        job_dir = os.path.join(self.results_folder, job_id)
                        if os.path.exists(job_dir):
                            shutil.rmtree(job_dir)
                        del self.jobs[job_id]
                except (ValueError, KeyError):
                    continue 