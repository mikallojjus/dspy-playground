# Dual Ollama Instance Setup

This project uses **two separate Ollama instances** to eliminate memory conflicts when switching between LLM and embedding models.

## Problem This Solves

**Issue:** Ollama has a pinned memory leak bug when switching models after concurrent operations. When running claim extraction with `batch_size=3`, Ollama creates multiple pinned memory mappings that aren't properly cleaned up, causing the embedding model to fail with "resource already mapped" errors.

**Solution:** Run two isolated Ollama instances in Docker containers:
- **Instance 1 (port 11434):** Qwen 2.5 7B for claim extraction and entailment
- **Instance 2 (port 11435):** nomic-embed-text for embeddings

Each instance has its own CUDA context, preventing memory conflicts.

## Prerequisites

- Docker Desktop for Windows installed
- NVIDIA GPU with updated drivers
- At least 10GB free VRAM (RTX 3090/4090 recommended)

## Quick Start

### 1. Stop any existing Ollama service
```bash
# If you have Ollama running outside Docker, stop it
ollama stop
# Or kill the process in Task Manager
```

### 2. Run the setup script
```bash
cd X:\work\dspy-playground
.\scripts\setup_ollama_instances.bat
```

This script will:
- Start two Ollama containers with GPU support
- Pull `qwen2.5:7b-instruct-q4_0` into instance 1 (port 11434)
- Pull `nomic-embed-text` into instance 2 (port 11435)
- Verify both instances are ready

**First-time run:** Model downloads may take 5-10 minutes.

### 3. Verify setup
```bash
# Check both instances are running
docker ps

# Should see:
# ollama-qwen      -> 11434:11434
# ollama-embedding -> 11435:11434

# Test LLM instance
curl http://localhost:11434/api/tags

# Test embedding instance
curl http://localhost:11435/api/tags
```

### 4. Run your extraction pipeline
```bash
uv run python -m src.cli.extract_claims --podcast-id 9 --limit 1
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Python Application                             │
│                                                 │
│  ┌──────────────────┐    ┌──────────────────┐  │
│  │ DSPy Models      │    │ Embedding Service│  │
│  │ (Qwen 2.5 7B)    │    │ (nomic-embed)    │  │
│  └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │             │
└───────────┼───────────────────────┼─────────────┘
            │                       │
    Port 11434               Port 11435
            │                       │
┌───────────▼─────────┐    ┌────────▼────────────┐
│ Docker Container 1  │    │ Docker Container 2  │
│ ollama-qwen         │    │ ollama-embedding    │
│                     │    │                     │
│ Model: Qwen 2.5 7B  │    │ Model: nomic-embed  │
│ VRAM: ~6GB          │    │ VRAM: ~0.8GB        │
│                     │    │                     │
│  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │ CUDA Context 1│  │    │  │ CUDA Context 2│  │
│  └───────┬───────┘  │    │  └───────┬───────┘  │
└──────────┼──────────┘    └──────────┼──────────┘
           │                          │
           └──────────┬───────────────┘
                      │
              ┌───────▼────────┐
              │ RTX 4090 GPU   │
              │ 24GB VRAM      │
              │ (~7GB used)    │
              └────────────────┘
```

## Configuration Files

### docker-compose.yml
Defines both Ollama services with GPU support, health checks, and persistent volumes.

### .env
```ini
OLLAMA_URL=http://localhost:11434                # LLM instance
OLLAMA_EMBEDDING_URL=http://localhost:11435      # Embedding instance
```

### Code Changes
- `src/config/settings.py`: Added `ollama_embedding_url` field
- `src/infrastructure/embedding_service.py`: Uses `settings.ollama_embedding_url` instead of `settings.ollama_url`

## Management Commands

### Start instances
```bash
docker-compose up -d
```

### Stop instances
```bash
docker-compose down
```

### Restart instances
```bash
docker-compose restart
```

### View logs
```bash
# Both instances
docker-compose logs -f

# Specific instance
docker logs -f ollama-qwen
docker logs -f ollama-embedding
```

### Check resource usage
```bash
# See VRAM usage
nvidia-smi

# Check Docker container stats
docker stats ollama-qwen ollama-embedding
```

### Shell into container
```bash
docker exec -it ollama-qwen bash
docker exec -it ollama-embedding bash
```

## Troubleshooting

### Models not loading / "model not found" errors

**Cause:** Models weren't pulled into the correct instance.

**Fix:**
```bash
# Pull into LLM instance (port 11434)
docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_0

# Pull into embedding instance (port 11435)
docker exec ollama-embedding ollama pull nomic-embed-text

# Verify
docker exec ollama-qwen ollama list
docker exec ollama-embedding ollama list
```

### "Connection refused" errors

**Cause:** Containers not running or not ready yet.

**Fix:**
```bash
# Check container status
docker ps

# Restart if needed
docker-compose restart

# Wait 30 seconds for startup, then test
curl http://localhost:11434/api/tags
curl http://localhost:11435/api/tags
```

### Out of memory errors (still happening)

**Cause:** Both models trying to use GPU simultaneously with large contexts.

**Fix 1 - Reduce context size:**
```ini
# In .env
OLLAMA_NUM_CTX=8192  # Reduced from 16384
```

**Fix 2 - Use smaller quantization:**
```bash
docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_k_m  # Smaller than q4_0
```

**Fix 3 - Sequential processing:**
```ini
# In .env
PARALLEL_BATCH_SIZE=1  # Process one chunk at a time
```

### Docker Desktop GPU support not working

**Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails:
1. Update NVIDIA drivers (581.15+ recommended)
2. Update Docker Desktop to latest version
3. Enable WSL2 integration in Docker Desktop settings
4. Restart Docker Desktop

### Containers stuck in "starting" state

**Cause:** GPU busy with another process.

**Fix:**
```bash
# Check what's using GPU
nvidia-smi

# Kill any ollama processes outside Docker
taskkill /F /IM ollama.exe

# Restart containers
docker-compose restart
```

## Performance Notes

**VRAM Usage:**
- Qwen 2.5 7B (16k context): ~6GB
- nomic-embed-text: ~0.8GB
- **Total: ~7GB** (17GB free on RTX 4090)

**Processing Speed:**
- Claim extraction: ~12-15 seconds per episode (9 chunks, batch_size=3)
- Embedding generation: ~5-10 seconds for 19 chunks
- **No freezing, no OOM errors** ✓

**Compared to single instance:**
- **Stability:** 100% success rate (vs 0% with single instance)
- **Speed:** Same (no slowdown from dual instances)
- **VRAM overhead:** Minimal (~200MB extra for second container)

## Why This Works

1. **Isolated CUDA contexts:** Each container maintains its own CUDA context with separate memory mappings
2. **No model switching:** LLM always uses instance 1, embeddings always use instance 2
3. **Concurrent operations safe:** `batch_size=3` works perfectly because all parallel calls go to the same instance
4. **Pinned memory isolated:** Each instance manages its own pinned memory pool without conflicts

## Reverting to Single Instance

If you need to go back to single Ollama instance:

1. Stop containers: `docker-compose down`
2. In `.env`, set: `OLLAMA_EMBEDDING_URL=http://localhost:11434` (same as OLLAMA_URL)
3. Start native Ollama: `ollama serve`
4. Pull models: `ollama pull qwen2.5:7b-instruct-q4_0 && ollama pull nomic-embed-text`

**Note:** Single instance will still have the pinned memory bug. Use `PARALLEL_BATCH_SIZE=1` to avoid it.

## References

- [Ollama Issue #12580](https://github.com/ollama/ollama/issues/12580) - Memory layout allocation failures
- [Ollama Issue #8985](https://github.com/ollama/ollama/issues/8985) - Model switching hangs
- [Ollama Issue #10597](https://github.com/ollama/ollama/issues/10597) - Progressive memory leak
- [DSPy asyncify docs](https://dspy.ai/api/utils/asyncify/) - Thread-safe DSPy execution

## Support

For issues specific to this setup, check:
1. Container logs: `docker-compose logs`
2. Application logs: `logs/extraction_*.log`
3. GPU status: `nvidia-smi`
4. Docker status: `docker ps` and `docker stats`
