'''
start example dev server
'''
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT
FRONTEND_DIR = ROOT / "frontend"

procs: list[subprocess.Popen] = []


def init_env(dir: Path) -> None:
    env_file = dir / ".env"
    example_file = dir / ".env.example"
    if not env_file.exists() and example_file.exists():
        shutil.copy2(example_file, env_file)
        print(f"Copied {example_file.relative_to(ROOT)} -> {env_file.relative_to(ROOT)}")


def cleanup(*_: object) -> None:
    print("\nStopping all services...")
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("All services stopped.")


def main() -> None:
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    init_env(BACKEND_DIR)
    init_env(FRONTEND_DIR)

    print("Starting backend (examples/main.py on port 2024)...")
    backend = subprocess.Popen(
        ["uv", "run", "python", "-m", "examples.main"],
        cwd=str(BACKEND_DIR),
    )
    procs.append(backend)

    print("Starting frontend (pnpm dev)...")
    frontend = subprocess.Popen(
        ["pnpm", "dev"],
        cwd=str(FRONTEND_DIR),
    )
    procs.append(frontend)

    print()
    print("Backend  -> http://localhost:2024")
    print("Frontend -> http://localhost:5173")
    print()
    print("Press Ctrl+C to stop all services.")

    try:
        backend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
