"""
Colab launcher for FastAPI + React.
If React build fails, server still starts and serves fallback UI.
"""
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(".")
FRONTEND = os.path.join(ROOT, "frontend")


def run(cmd, *, cwd=None, env=None, check=True):
    print("▶", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def try_frontend_build() -> bool:
    run(["node", "--version"], check=False)
    run(["npm", "--version"], check=False)

    lock_file = os.path.join(FRONTEND, "package-lock.json")
    if os.path.exists(lock_file):
        if run(["npm", "ci", "--legacy-peer-deps"], cwd=FRONTEND, check=False).returncode != 0:
            return False
    else:
        if run(["npm", "install", "--legacy-peer-deps"], cwd=FRONTEND, check=False).returncode != 0:
            return False

    build_env = {
        **os.environ,
        "CI": "false",
        "GENERATE_SOURCEMAP": "false",
        "NODE_OPTIONS": os.environ.get("NODE_OPTIONS", "--max_old_space_size=4096"),
    }
    if run(["npm", "run", "build"], cwd=FRONTEND, env=build_env, check=False).returncode == 0:
        return True

    print("⚠ npm build failed; retrying after cache clean...")
    run(["npm", "cache", "clean", "--force"], cwd=FRONTEND, check=False)
    if run(["npm", "run", "build"], cwd=FRONTEND, env=build_env, check=False).returncode == 0:
        return True

    print("⚠ npm build failed again; retrying after clean install...")
    shutil.rmtree(os.path.join(FRONTEND, "node_modules"), ignore_errors=True)
    if os.path.exists(lock_file):
        os.remove(lock_file)
    if run(["npm", "install", "--legacy-peer-deps"], cwd=FRONTEND, check=False).returncode != 0:
        return False
    return run(["npm", "run", "build"], cwd=FRONTEND, env=build_env, check=False).returncode == 0


# 1) Python deps
run([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn[standard]", "python-multipart",
    "datasets>=2.14", "huggingface-hub>=0.16",
], check=True)

# 2) Frontend build (best effort)
print("▶ Installing npm deps...")
print("▶ Building React app (takes ~60s on first run)...")
built = try_frontend_build()
if built:
    print("✓ React build complete.")
else:
    print("⚠ React build failed after retries. Continuing with FastAPI fallback UI.")

# 3) Start FastAPI
print("▶ Starting FastAPI server…")
subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn",
        "backend.server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "warning",
    ],
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

# 4) Tunnel
ip = urllib.request.urlopen("https://ipv4.icanhazip.com").read().decode().strip()
print()
print("=" * 55)
print(f"  TUNNEL PASSWORD → {ip}")
print("  Paste this IP at the loca.lt splash page to enter.")
print("=" * 55)
print()

os.execlp("npx", "npx", "localtunnel", "--port", "8000")
