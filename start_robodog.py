"""
Single entry point for the robot.

Run ONLY this file (`python3 start_robodog.py`). It will:

  1. Check whether a trained face model already exists.
  2. If not (or if you ask it to), walk you through capturing faces and
     training - the same steps capturefaces.py / trainfaces.py do, just
     chained together with one prompt instead of two manual runs.
  3. Launch main.py - the voice + servo control runtime - automatically.

This does not replace capturefaces.py / trainfaces.py / main.py; it just
calls them in the right order so you don't have to run three separate
commands every time.
"""

import os
import subprocess
import sys

MODEL_PATH = os.path.join("models", "trainer.yml")
LABELS_PATH = os.path.join("models", "labels.npy")


def run_script(script):

    print()
    print("=" * 60)
    print(f"RUNNING: {script}")
    print("=" * 60)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"ERROR: {script} failed")
        sys.exit(1)


def model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)


def offer_face_setup():

    print()

    if model_exists():
        prompt = "A trained face model already exists. Capture/train another person now? (y/N): "
    else:
        print("No trained face model found yet.")
        prompt = "Capture and train faces now? (y/N): "

    if input(prompt).strip().lower() != "y":
        return

    while True:
        run_script("capturefaces.py")
        if input("Add another person? (y/N): ").strip().lower() != "y":
            break

    run_script("trainfaces.py")


def main():

    print("ROBODOG STARTUP")

    offer_face_setup()

    if not model_exists():
        print()
        print("WARNING: no trained face model found - 'recognize me' will")
        print("say unknown until you run this again and train one.")

    print()
    print("Starting main.py ...")
    print()

    # os.execv REPLACES this process with main.py, rather than spawning a
    # child process. That means Ctrl+C, and the atexit cleanup already
    # registered inside main.py (legs.shutdown(), cleanup_camera()),
    # behave exactly as if you'd run `python3 main.py` directly.
    os.execv(sys.executable, [sys.executable, "main.py"])


if __name__ == "__main__":
    main()
