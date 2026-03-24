"""
Paste this entire file as a single Colab code cell, then Run it.
It replaces the old streamlit launch with the new React + FastAPI stack.
"""
import os, subprocess, time, urllib.request, sys

ROOT = os.path.abspath(".")

# ── 1. Install Python deps ────────────────────────────────────────
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn[standard]", "python-multipart",
    "datasets>=2.14", "huggingface-hub>=0.16",
], check=True)

# ── 2. Build React frontend ───────────────────────────────────────
print("▶ Installing npm deps...")
subprocess.run(
    ["npm", "install", "--legacy-peer-deps", "--silent"],
    cwd=os.path.join(ROOT, "frontend"), check=True,
)
print("▶ Building React app (takes ~60s on first run)...")
subprocess.run(
    ["npm", "run", "build"],
    cwd=os.path.join(ROOT, "frontend"), check=True,
    env={**os.environ, "CI": "false"},   # don't treat warnings as errors
)
print("✓ React build complete.")

# ── 3. Start FastAPI (API + React static) on port 8000 ───────────
print("▶ Starting FastAPI server…")
subprocess.Popen(
    [sys.executable, "-m", "uvicorn",
     "backend.server:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--log-level", "warning"],
    cwd=ROOT,
    env={
        **os.environ,
        "RUST_BINARY": os.path.join(ROOT, "target", "release", "csrre"),
        "PYTHONPATH": os.pathsep.join([
            os.path.join(ROOT, "lcs", "training"),
            os.path.join(ROOT, "lcs", "induction"),
        ]),
    },
)
time.sleep(4)

# ── 4. Open LocalTunnel ───────────────────────────────────────────
ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip()
print()
print("=" * 55)
print(f"  TUNNEL PASSWORD → {ip}")
print("  Paste this IP at the loca.lt splash page to enter.")
print("=" * 55)
print()

os.execlp("npx", "npx", "localtunnel", "--port", "8000")
