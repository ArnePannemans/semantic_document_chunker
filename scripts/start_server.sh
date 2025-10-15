#!/bin/bash
# Startup script for the Unsloth serving API

echo "Starting Semantic Chunking API Server..."
echo "=========================================="
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Warning: Virtual environment not found at venv/bin/activate"
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q fastapi uvicorn[standard] requests

echo ""
echo "Starting server..."
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python -m src.api.server

