FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY cloud/ ./cloud/
COPY cli/ ./cli/

# Create directories for models and data
RUN mkdir -p /models /data /logs

# Set environment variables
ENV PYTHONPATH=/app
ENV USE_GPU=true
ENV MODEL_PATH=/models

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "-m", "src.api.main"]