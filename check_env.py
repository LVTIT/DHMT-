"""
check_env.py - Quick environment check for the Human Motion Capture project.

Run:
    python check_env.py
"""

import importlib
from pathlib import Path


REQUIRED_MODULES = [
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("websockets", "websockets"),
]


def main() -> int:
    root = Path(__file__).resolve().parent
    ok = True

    print("[CHECK] Human Motion Capture environment")
    print(f"[CHECK] Project root: {root}")

    model_path = root / "pose_landmarker.task"
    if model_path.exists():
        print(f"[OK] MediaPipe model: {model_path.name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("[FAIL] Missing pose_landmarker.task")
        ok = False

    for module_name, package_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {package_name}: {version}")
        except Exception as exc:
            print(f"[FAIL] {package_name}: {exc}")
            ok = False

    static_index = root / "static" / "index.html"
    if static_index.exists():
        print("[OK] WebGL viewer: static/index.html")
    else:
        print("[FAIL] Missing static/index.html")
        ok = False

    print("[CHECK] Result:", "ready" if ok else "needs attention")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
