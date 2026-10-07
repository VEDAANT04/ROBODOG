import cv2
import os
import time
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

TARGET_IMAGES = 30

IMAGE_SIZE = 200

CAPTURE_INTERVAL = 0.4


# =========================================================
# CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

camera.set(
    cv2.CAP_PROP_FPS,
    30
)

if not camera.isOpened():

    print(
        "ERROR: Cannot open camera"
    )

    exit()


# =========================================================
# FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(

    cv2.data.haarcascades
    +
    "haarcascade_frontalface_default.xml"

)


# =========================================================
# GET NAME
# =========================================================

name = input(
    "Enter person's name: "
).lower().strip()

if not name:

    print(
        "Name cannot be empty"
    )

    exit()


dataset_path = os.path.join(

    "dataset",

    name

)

os.makedirs(

    dataset_path,

    exist_ok=True

)


# =========================================================
# QUALITY CHECK
# =========================================================

def is_good_face(face):

    brightness = np.mean(
        face
    )

    if (
        brightness < 40
        or brightness > 220
    ):

        return False

    blur_value = cv2.Laplacian(

        face,

        cv2.CV_64F

    ).var()

    if blur_value < 80:

        return False

    return True


# =========================================================
# CAPTURE
# =========================================================

print()

print(
    "FACE CAPTURE STARTED"
)

print(
    f"Target: {TARGET_IMAGES} images"
)

print(
    "Move your face slowly to different angles"
)

print(
    "Press ESC to stop"
)

count = 0

last_capture = 0


while count < TARGET_IMAGES:

    ret, frame = camera.read()

    if not ret:

        continue

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )

    faces = face_detector.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=6,

        minSize=(80, 80)

    )

    if len(faces) > 0:

        x, y, w, h = max(

            faces,

            key=lambda face:
            face[2] * face[3]

        )

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )

        current_time = time.time()

        if (
            current_time
            - last_capture
            >= CAPTURE_INTERVAL
        ):

            face_img = gray[
                y:y+h,
                x:x+w
            ]

            face_img = cv2.resize(

                face_img,

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )

            )

            face_img = cv2.equalizeHist(
                face_img
            )

            if is_good_face(face_img):

                count += 1

                filename = os.path.join(

                    dataset_path,

                    f"{count:03d}.jpg"

                )

                cv2.imwrite(

                    filename,

                    face_img

                )

                print(

                    f"Captured "
                    f"{count}/{TARGET_IMAGES}"

                )

                last_capture = current_time

    cv2.putText(

        frame,

        f"Images: {count}/{TARGET_IMAGES}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )

    cv2.imshow(

        "RoboDog Face Capture",

        frame

    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:

        break


camera.release()

cv2.destroyAllWindows()

print()

print(
    f"Capture complete: "
    f"{count} images saved"
)

print(
    "Next: run trainfaces.py"
)