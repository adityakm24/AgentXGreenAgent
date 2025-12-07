FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements (manually defined for now since we use uv locals)
# Alternatively, we can generate requirements.txt
COPY pyproject.toml .
# Make a temporary requirements.txt for pip
RUN echo "fastapi\nuvicorn\nrequests\npydantic\npython-dotenv\nopenai" > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .

# Create data directory
RUN mkdir -p data

# Note: The database 'jurist.db' needs to be populated. 
# In a real deployed container, we might want to mount it or ingest on build.
# For this submission, we'll copy the ingestion script and allow running it.
# Ideally, we COPY the populated DB if it's small enough, or download data.
# We will assume data is mounted or we run ingestion entrypoint.

EXPOSE 8000

# Default command
CMD ["python", "main.py"]
