FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y --no-install-recommends curl git && \
  rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . /app/

# Install Python dependencies
RUN uv sync --frozen

# Create logs directory
RUN mkdir -p /app/logs

# Railway sets PORT environment variable
ENV PORT=8000

# Expose port (Railway will override with its own PORT)
EXPOSE 8000

# Start command (Railway will use this if no Procfile/railway.toml)
CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
