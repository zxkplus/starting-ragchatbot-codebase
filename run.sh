#!/bin/bash

# Create necessary directories
mkdir -p docs

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Clear ALL proxy env vars to avoid interfering with TUN-mode proxy (Clash etc.)
# The TUN-mode virtual NIC handles routing transparently via DNS fake IP
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY ftp_proxy FTP_PROXY
unset no_proxy NO_PROXY

# Use Hugging Face mirror for downloading models in restricted network environments
export HF_ENDPOINT=https://hf-mirror.com

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000
