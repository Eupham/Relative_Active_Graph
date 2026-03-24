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
echo "▶ Node/NPM versions:"
node --version
npm --version

if [ -f package-lock.json ]; then
  npm ci --legacy-peer-deps
else
  npm install --legacy-peer-deps
fi

if ! CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build; then
  echo "⚠ First frontend build failed; retrying after npm cache clean..."
  npm cache clean --force || true
fi
if ! CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build; then
  echo "⚠ Second frontend build failed; retrying after clean install..."
  rm -rf node_modules package-lock.json
  npm install --legacy-peer-deps
  CI=false GENERATE_SOURCEMAP=false NODE_OPTIONS=${NODE_OPTIONS:---max_old_space_size=4096} npm run build
fi
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
