FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for runtime
RUN mkdir -p /app/visualizations /app/ml/models

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
