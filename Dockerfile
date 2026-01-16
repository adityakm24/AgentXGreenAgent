FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY src/ /app/src/
COPY data_repo/ /app/data_repo/
COPY pyproject.toml /app/

# Install python dependencies
# Using pip directly for simplicity in submission container, referencing pyproject.toml libs
RUN pip install --no-cache-dir \
    fastapi[standard] \
    uvicorn[standard] \
    httpx \
    rich \
    jsonlines \
    langchain \
    langchain-openai \
    langgraph \
    pydantic \
    pydantic-settings \
    python-dotenv \
    # Helper libs for runner scripts (optional but good for local testing inside container)
    tomli \
    tomli-w \
    requests

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port (default 9009)
EXPOSE 9009

# Run the server
ENTRYPOINT ["python", "src/green_agent/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
