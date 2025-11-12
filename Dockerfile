FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
  apt-get install -y curl git && \
  rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . /app/

# Install Python dependencies
RUN uv sync

# Activate venv and run
CMD ["/bin/bash", "-c", "source .venv/bin/activate && exec bash"]
