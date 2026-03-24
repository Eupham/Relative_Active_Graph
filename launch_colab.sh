#!/bin/bash
# Colab launcher: builds React UI and starts FastAPI (port 8000) + LocalTunnel.
# Run from the repo root: bash launch_colab.sh

set -e

echo ""
echo "═══════════════════════════════════════════════════"
echo "  RAG Engine — Colab Launcher"
echo "═══════════════════════════════════════════════════"

# 1. Python deps
echo "▶ Installing Python dependencies..."
pip install -q fastapi uvicorn[standard] python-multipart

# 2. Build React frontend
echo "▶ Building React frontend..."
cd frontend
npm install --legacy-peer-deps --silent
npm run build
cd ..
echo "✓ React build complete."

# 3. Start FastAPI (serves API + React build)
echo "▶ Starting FastAPI server on port 8000..."
nohup uvicorn backend.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
sleep 4

# 4. Tunnel
IP=$(curl -s https://ipv4.icanhazip.com)
echo ""
echo "═══════════════════════════════════════════════════"
echo "  TUNNEL PASSWORD: ${IP}"
echo "  (Enter this IP at the loca.lt splash page)"
echo "═══════════════════════════════════════════════════"
echo ""

npx localtunnel --port 8000
