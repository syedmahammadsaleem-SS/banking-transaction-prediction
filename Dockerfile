FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     libgomp1     && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/
COPY main_pipeline.py .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs reports/figures

# Expose ports
EXPOSE 5000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3     CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Default command
CMD ["python", "app/flask/app.py"]
