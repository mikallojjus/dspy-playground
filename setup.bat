@echo off
REM Docker Setup Script for DSPy Playground
REM This script sets up two Ollama instances (qwen + embedding) and a reranker service

echo ========================================
echo DSPy Playground Docker Setup
echo ========================================
echo.
echo This script will:
echo - Start Docker Compose services
echo - Pull qwen2.5:7b-instruct-q4_0 to ollama-qwen
echo - Pull nomic-embed-text to ollama-embedding
echo - Verify all services are ready
echo.

REM Start Docker Compose
echo [1/4] Starting Docker Compose services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker Compose
    exit /b 1
)
echo Docker Compose started successfully
echo.

REM Wait for services to be healthy
echo [2/4] Waiting for services to be healthy...
echo This may take 30-60 seconds...
timeout /t 30 /nobreak >nul
echo.

REM Pull qwen model to ollama-qwen (port 11434)
echo [3/4] Pulling qwen2.5:7b-instruct-q4_0 to ollama-qwen...
docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_0
if %errorlevel% neq 0 (
    echo WARNING: Failed to pull qwen model. You may need to pull it manually.
    echo Run: docker exec ollama-qwen ollama pull qwen2.5:7b-instruct-q4_0
) else (
    echo Qwen model pulled successfully
)
echo.

REM Pull embedding model to ollama-embedding (port 11435)
echo [4/4] Pulling nomic-embed-text to ollama-embedding...
docker exec ollama-embedding ollama pull nomic-embed-text
if %errorlevel% neq 0 (
    echo WARNING: Failed to pull embedding model. You may need to pull it manually.
    echo Run: docker exec ollama-embedding ollama pull nomic-embed-text
) else (
    echo Embedding model pulled successfully
)
echo.

REM Verify services
echo ========================================
echo Verifying services...
echo ========================================
echo.

echo Checking ollama-qwen (port 11434)...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] ollama-qwen is responding
) else (
    echo [WARNING] ollama-qwen is not responding yet
)

echo Checking ollama-embedding (port 11435)...
curl -s http://localhost:11435/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] ollama-embedding is responding
) else (
    echo [WARNING] ollama-embedding is not responding yet
)

echo Checking reranker (port 8080)...
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] reranker is responding
) else (
    echo [WARNING] reranker is not responding yet
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Service endpoints:
echo - Ollama Qwen:     http://localhost:11434
echo - Ollama Embedding: http://localhost:11435
echo - Reranker:        http://localhost:8080
echo.
echo To stop services: docker-compose down
echo To view logs: docker-compose logs -f
echo.
pause
