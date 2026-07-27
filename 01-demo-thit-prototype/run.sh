#!/bin/bash

# Apollo Hospital Voice AI Assistant v2.0
# Startup script with Redis and seed data support

set -e

echo "=========================================="
echo "  Apollo Hospital Voice AI Assistant"
echo "  v2.0 - Hackathon Demo"
echo "=========================================="
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set."
    echo "You may need to set it for LLaMA model access:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo ""
fi

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check GPU availability
echo ""
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {\"GPU - \" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Check Redis
echo ""
echo "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "Redis is running!"
        REDIS_CONNECTED=true
    else
        echo "Redis is installed but not running."
        echo "Starting Redis..."
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes
            sleep 1
            if redis-cli ping &> /dev/null; then
                echo "Redis started successfully!"
                REDIS_CONNECTED=true
            else
                echo "WARNING: Could not start Redis. Using in-memory fallback."
                REDIS_CONNECTED=false
            fi
        else
            echo "WARNING: Redis server not found. Using in-memory fallback."
            REDIS_CONNECTED=false
        fi
    fi
else
    echo "WARNING: Redis not installed. Using in-memory fallback."
    echo "To install Redis:"
    echo "  macOS: brew install redis"
    echo "  Ubuntu: sudo apt install redis-server"
    REDIS_CONNECTED=false
fi

# Seed data if Redis is connected
if [ "$REDIS_CONNECTED" = true ]; then
    echo ""
    echo "Seeding hospital data..."
    python3 scripts/seed_data.py --clear 2>/dev/null || echo "Seed script not found or failed. Continuing..."
fi

echo ""
echo "=========================================="
echo "  Starting server on http://localhost:8000"
echo "=========================================="
echo ""
echo "Endpoints:"
echo "  - Main UI:    http://localhost:8000"
echo "  - Admin:      http://localhost:8000/admin"
echo "  - Health:     http://localhost:8000/health"
echo "  - API Docs:   http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
