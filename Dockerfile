# Use an official Python image as the base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Create required directory structure
RUN mkdir -p /app/web_app/uploads \
    /app/web_app/results \
    /app/web_app/app/models \
    /app/web_app/app/templates \
    /app/web_app/app/static \
    /app/examples

# Copy application code
COPY . .

# Make sure templates exist
COPY web_app/app/templates/enhanced.html /app/web_app/app/templates/
COPY web_app/app/templates/simple.html /app/web_app/app/templates/

# Copy example files if they exist
COPY examples/* /app/examples/

# Copy the advanced_processor module
COPY advanced_processor.py /app/

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8080

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "fastapi_enhanced:app", "--host", "0.0.0.0", "--port", "8080"]