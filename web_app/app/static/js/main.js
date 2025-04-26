document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const estimatedTime = document.getElementById('estimated-time');
    const statusMessage = document.getElementById('status-message');
    const fpsCounter = document.getElementById('fps-counter');
    const resultSection = document.getElementById('result-section');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        progressSection.classList.remove('hidden');
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            // Start polling for progress
            const progressInterval = setInterval(async () => {
                const progressResponse = await fetch('/progress');
                const progressData = await progressResponse.json();
                
                progressBar.style.width = `${progressData.progress}%`;
                progressPercentage.textContent = `${progressData.progress}%`;
                estimatedTime.textContent = `Estimated time remaining: ${progressData.estimated_time}`;
                statusMessage.textContent = progressData.status_message;
                fpsCounter.textContent = `Processing speed: ${progressData.fps.toFixed(2)} FPS`;

                if (progressData.completed) {
                    clearInterval(progressInterval);
                    resultSection.classList.remove('hidden');
                    
                    if (progressData.result_path) {
                        document.getElementById('processed-video').src = progressData.result_path;
                    }
                }
            }, 1000);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the video');
        }
    });
});
