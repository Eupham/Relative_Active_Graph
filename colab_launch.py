"""
Paste this entire file as a single Colab code cell, then Run it.
It replaces the old streamlit launch with the new React + FastAPI stack.
"""
import os, shutil, subprocess, time, urllib.request, sys

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
frontend_dir = os.path.join(ROOT, "frontend")
build_env = {**os.environ, "CI": "false", "GENERATE_SOURCEMAP": "false"}
try:
    subprocess.run(
        ["npm", "run", "build"],
        cwd=frontend_dir, check=True,
        env=build_env,   # don't treat warnings as errors; reduce memory usage
    )
except subprocess.CalledProcessError:
    # Colab can keep stale/incompatible node_modules around between sessions.
    print("⚠ npm build failed; retrying after clean install...")
    shutil.rmtree(os.path.join(frontend_dir, "node_modules"), ignore_errors=True)
    lock_file = os.path.join(frontend_dir, "package-lock.json")
    if os.path.exists(lock_file):
        os.remove(lock_file)
    subprocess.run(
        ["npm", "install", "--legacy-peer-deps"],
        cwd=frontend_dir, check=True,
    )
    subprocess.run(
        ["npm", "run", "build"],
        cwd=frontend_dir, check=True,
        env=build_env,
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
