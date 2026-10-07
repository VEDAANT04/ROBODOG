import subprocess
import sys


def run_script(script):

    print()

    print("=" * 60)

    print(
        f"RUNNING: {script}"
    )

    print("=" * 60)

    result = subprocess.run(

        [
            sys.executable,
            script
        ]

    )

    if result.returncode != 0:

        print(

            f"ERROR: {script} failed"

        )

        sys.exit(1)


print(

    "ROBODOG FACE RECOGNITION WORKFLOW"

)


# STEP 1

run_script(

    "capturefaces.py"

)


# STEP 2

run_script(

    "trainfaces.py"

)


# STEP 3

print()

print(

    "VERIFYING FACE..."

)

from facerecognizer import (

    init_camera,

    recognize_face,

    cleanup_camera

)

try:

    if init_camera():

        name = recognize_face()

        print()

        print(

            f"Recognition result: {name}"

        )

    else:

        print(

            "Camera could not start"

        )

finally:

    cleanup_camera()


print()

print(

    "WORKFLOW COMPLETE"

)