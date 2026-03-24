#!/bin/bash
# Colab launcher: builds React UI and starts FastAPI (port 8000) + LocalTunnel.
# Run from the repo root: bash launch_colab.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "frontend/package.json" ]; then
  echo "ERROR: frontend/package.json not found."
  echo "Run this script from the repository root or keep the script in the repo root."
  exit 1
fi

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
echo "▶ Node/NPM versions:"
node --version
npm --version

BUILD_OK=0
if [ -f package-lock.json ]; then
  npm ci --legacy-peer-deps || BUILD_OK=1
else
  npm install --legacy-peer-deps || BUILD_OK=1
fi

if [ "$BUILD_OK" -eq 0 ] && CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build; then
  BUILD_OK=0
else
  echo "⚠ First frontend build failed; retrying after npm cache clean..."
  npm cache clean --force || true
  if CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build; then
    BUILD_OK=0
  else
    echo "⚠ Second frontend build failed; retrying after clean install..."
    rm -rf node_modules package-lock.json
    npm install --legacy-peer-deps || BUILD_OK=1
    if CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build; then
      BUILD_OK=0
    else
      BUILD_OK=1
    fi
  fi
fi
cd ..
if [ "$BUILD_OK" -eq 0 ]; then
  echo "✓ React build complete."
else
  echo "⚠ React build failed after retries; continuing with FastAPI fallback UI."
fi

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
