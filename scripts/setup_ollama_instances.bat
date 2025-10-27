@echo off
REM Setup script for dual Ollama instances with GPU support
REM This script starts two isolated Ollama containers and pulls required models

echo ==========================================
echo Setting up Dual Ollama Instances
echo ==========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [1/5] Stopping any existing Ollama containers...
docker-compose down 2>nul

echo.
echo [2/5] Starting Ollama instances (this may take 30-60 seconds)...
docker-compose up -d

echo.
echo [3/5] Waiting for services to be ready...
timeout /t 15 /nobreak >nul

REM Wait for ollama-qwen to be ready
:wait_qwen
echo Checking ollama-qwen readiness...
docker exec ollama-qwen curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo   Not ready yet, waiting 5 seconds...
    timeout /t 5 /nobreak >nul
    goto wait_qwen
)
echo   ✓ ollama-qwen ready on port 11434

REM Wait for ollama-embedding to be ready
:wait_embedding
echo Checking ollama-embedding readiness...
docker exec ollama-embedding curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo   Not ready yet, waiting 5 seconds...
    timeout /t 5 /nobreak >nul
    goto wait_embedding
)
echo   ✓ ollama-embedding ready on port 11435

echo.
echo [4/5] Pulling models into respective instances...
echo.
echo   Pulling qwen2.5:7b-instruct-q4_0 into ollama-qwen (port 11434)...
echo   This may take several minutes on first run...
docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_0

echo.
echo   Pulling nomic-embed-text into ollama-embedding (port 11435)...
docker exec ollama-embedding ollama pull nomic-embed-text

echo.
echo [5/5] Verifying setup...
echo.
echo   Checking ollama-qwen models:
docker exec ollama-qwen ollama list
echo.
echo   Checking ollama-embedding models:
docker exec ollama-embedding ollama list

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Ollama instances are running:
echo   - Qwen 2.5 7B:        http://localhost:11434
echo   - Embedding model:    http://localhost:11435
echo.
echo To stop instances:    docker-compose down
echo To view logs:         docker-compose logs -f
echo To restart:           docker-compose restart
echo.
pause
