# DSPy Playground

**DSPy Playground** is an intelligent claim and quote extraction pipeline for podcast transcripts, powered by DSPy. The system automatically extracts factual claims from podcast episodes, finds supporting quotes from transcripts using semantic search and reranking, validates entailment relationships between claims and quotes, and filters out advertisement content—all while maintaining high accuracy through few-shot learning and LLM-as-judge metrics. Built for production use with PostgreSQL storage, pgvector similarity search, intelligent deduplication, and comprehensive evaluation datasets.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Install uv (Python Package Manager)](#1-install-uv-python-package-manager)
  - [2. Clone Repository](#2-clone-repository)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Configure Environment](#4-configure-environment)
  - [5. Docker Setup](#5-docker-setup)
  - [6. Install Models to Ollama](#6-install-models-to-ollama)
  - [7. Database Setup](#7-database-setup)
- [Usage](#usage)
  - [Process Episodes](#process-episodes)
  - [Train Models](#train-models)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- **Claim Extraction**: Automatically extract factual claims from podcast transcripts using DSPy-optimized LLMs
- **Quote Discovery**: Find supporting quotes using semantic search (pgvector) + reranking (BGE-reranker-large)
- **Entailment Validation**: Verify claim-quote relationships using trained entailment models
- **Ad Filtering**: Classify and filter advertisement claims with confidence-based thresholds
- **Smart Deduplication**: Multi-stage deduplication using embeddings, reranking, and string similarity
- **MIPROv2 Optimization**: Train models using state-of-the-art prompt optimization
- **Production Ready**: PostgreSQL storage, async processing, comprehensive logging, progress tracking

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.12+** installed
- **Docker** and **Docker Compose** installed
- **PostgreSQL database** (can be local or remote)

---

## Setup Guide

### 1. Install uv (Python Package Manager)

`uv` is a fast Python package installer and resolver. Install it for your platform:

#### **Windows**

Using PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or using pip:

```bash
pip install uv
```

#### **macOS**

Using Homebrew:

```bash
brew install uv
```

Or using curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### **Linux**

Using curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:

```bash
pip install uv
```

**Verify installation:**

```bash
uv --version
```

---

### 2. Install Dependencies

Install all Python dependencies using `uv`:

```bash
# Create virtual environment and install dependencies
uv sync

# On Windows (CMD):
.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

This will install all dependencies defined in `pyproject.toml`:

- DSPy 3.0+
- SQLAlchemy + psycopg2 (PostgreSQL)
- pgvector (vector similarity search)
- Rich (CLI formatting)
- Pydantic (configuration management)
- pytest (testing)

---

### 3. Configure Environment

Copy the example environment file and configure it:

```bash
# Copy example config
cp .env.example .env
```

---

### 5. Docker Setup

The project uses Docker Compose to run:

- **ollama-qwen**: Ollama instance for LLM inference (port 11434)
- **ollama-embedding**: Ollama instance for embeddings (port 11435)
- **reranker**: HuggingFace TEI reranker service (port 8080)

#### **Start Docker Services**

##### **Windows**

Using the provided setup script:

```bash
setup.bat
```

Or manually:

```bash
docker-compose up -d
```

##### **macOS/Linux**

Create a setup script or run manually:

```bash
# Start services
docker-compose up -d

# Pull models (see next section)
```

#### **Verify Services**

Check that all services are running:

```bash
# Check container status
docker-compose ps

# Check logs
docker-compose logs -f

# Test endpoints
curl http://localhost:11434/api/tags  # ollama-qwen
curl http://localhost:11435/api/tags  # ollama-embedding
curl http://localhost:8080/health      # reranker
```

#### **GPU Support (Optional)**

For GPU acceleration, ensure:

1. **NVIDIA drivers** are installed
2. **NVIDIA Container Toolkit** is installed:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

3. Verify GPU is available in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

### 6. Install Models to Ollama

Pull the required models to both Ollama instances:

#### **Pull LLM Model (ollama-qwen)**

```bash
# Qwen 2.5 7B 
docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_0

# OR Qwen 3 4B 
docker exec ollama-qwen ollama pull qwen3:4b
```

#### **Pull Embedding Model (ollama-embedding)**

```bash
docker exec ollama-embedding ollama pull nomic-embed-text
```

#### **Verify Models**

```bash
# List models in ollama-qwen
docker exec ollama-qwen ollama list

# List models in ollama-embedding
docker exec ollama-embedding ollama list
```

**Example output:**

```
NAME                            ID              SIZE    MODIFIED
qwen2.5:7b-instruct-q4_0       abc123def456    4.4 GB  2 minutes ago
nomic-embed-text:latest        789xyz012345    274 MB  1 minute ago
```

#### **Model Options**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `qwen2.5:7b-instruct-q4_0` | 4.4 GB | Medium | High | Recommended (best balance) |
| `qwen3:4b` | 2.3 GB | Fast | Medium | Quick experimentation |
| `llama3:8b-instruct-q4_0` | 4.7 GB | Medium | High | Alternative to Qwen |
| `mistral:7b-instruct-q4_0` | 4.1 GB | Medium | High | Alternative to Qwen |

**Update your `.env` file with the chosen model:**

```bash
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_0
```

---

## Usage

### Process Episodes

Process podcast episodes through the claim extraction pipeline:

```bash
# Process all unprocessed episodes
uv run python -m src.cli.process_episodes

# Process specific episode
uv run python -m src.cli.process_episodes --episode-id 123

# Process specific podcast, limit to 5 episodes
uv run python -m src.cli.process_episodes --podcast-id 9 --limit 5

# Continue on errors
uv run python -m src.cli.process_episodes --continue-on-error

# Reprocess all (ignore existing claims)
uv run python -m src.cli.process_episodes --force
```

**Output:**

- Progress bars with real-time status
- Per-episode statistics (claims, quotes, duplicates)
- Final summary with success/failure counts
- Logs saved to `logs/extraction_YYYYMMDD_HHMMSS.log`

---

### Train Models

Train or retrain the DSPy models using your evaluation datasets:

#### **Ad Classifier**

```bash
uv run python -m src.training.train_ad_classifier
```

#### **Claim Extractor (BootstrapFewShot)**

```bash
uv run python -m src.training.train_claim_extractor
```

#### **Claim Extractor (MIPROv2)**

Requires Anthropic API key in `.env`:

```bash
uv run python -m src.training.train_claim_extractor_mipro
```

#### **Entailment Validator (MIPROv2)**

```bash
uv run python -m src.training.train_entailment_validator_mipro
```

---

## Project Structure

```
dspy-playground/
├── src/
│   ├── cli/                    # Command-line interfaces
│   │   ├── process_episodes.py    # Main episode processing CLI
│   │   └── episode_query.py       # Episode query service
│   ├── config/                 # Configuration management
│   │   └── settings.py            # Pydantic settings
│   ├── database/               # Database models and utilities
│   │   └── models.py              # SQLAlchemy models
│   ├── deduplication/          # Claim deduplication logic
│   ├── dspy_models/            # DSPy model definitions
│   ├── extraction/             # Claim extraction
│   │   └── claim_extractor.py     # Main extractor
│   ├── infrastructure/         # Logging, caching, etc.
│   │   └── logger.py              # Rich-based logging
│   ├── metrics/                # Evaluation metrics
│   │   ├── ad_metrics.py          # Ad classification metrics
│   │   └── ...                    # Other metrics
│   ├── pipeline/               # End-to-end pipeline
│   │   └── extraction_pipeline.py # Main pipeline
│   ├── preprocessing/          # Text chunking and preprocessing
│   ├── scoring/                # Quote scoring and ranking
│   ├── search/                 # Semantic search
│   └── training/               # Model training scripts
│       ├── train_ad_classifier.py
│       ├── train_claim_extractor_mipro.py
│       └── train_entailment_validator_mipro.py
├── evaluation/                 # Evaluation datasets
│   ├── ad_train.json              # Ad classification training
│   ├── ad_val.json                # Ad classification validation
│   ├── claims_train.json          # Claim extraction training
│   ├── claims_val.json            # Claim extraction validation
│   ├── entailment_train.json      # Entailment training
│   └── entailment_val.json        # Entailment validation
├── models/                     # Trained model artifacts
│   ├── ad_classifier_v1.json
│   ├── claim_extractor_v1.json
│   └── entailment_validator_v1.json
├── logs/                       # Application logs
├── docker-compose.yml          # Docker services configuration
├── setup.bat                   # Windows setup script
├── pyproject.toml              # Python dependencies
├── .env.example                # Example environment config
└── README.md                   # This file
```

---

**Happy claim extracting! 🚀**
