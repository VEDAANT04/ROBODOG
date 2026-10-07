import cv2
import os
import time
import numpy as np


CAMERA = None

RECOGNIZER = None

LABELS = None

FACE_DETECTOR = None

SYSTEM_READY = False


# =========================================================
# INITIALIZE CAMERA
# =========================================================

def init_camera():

    global CAMERA

    if CAMERA is not None:

        return True

    try:

        CAMERA = cv2.VideoCapture(0)

        CAMERA.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        CAMERA.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        CAMERA.set(
            cv2.CAP_PROP_FPS,
            30
        )

        CAMERA.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not CAMERA.isOpened():

            print(
                "ERROR: Cannot open camera"
            )

            return False

        # Camera warmup

        for _ in range(10):

            CAMERA.read()

        print(
            "Camera ready"
        )

        return True

    except Exception as e:

        print(
            "Camera initialization error:",
            e
        )

        return False


# =========================================================
# LOAD FACE SYSTEM ONCE
# =========================================================

def init_face_system():

    global RECOGNIZER
    global LABELS
    global FACE_DETECTOR
    global SYSTEM_READY

    if SYSTEM_READY:

        return True

    try:

        trainer_path = (
            "models/trainer.yml"
        )

        labels_path = (
            "models/labels.npy"
        )

        if not os.path.exists(
            trainer_path
        ):

            print(
                "Face model not found"
            )

            return False

        if not os.path.exists(
            labels_path
        ):

            print(
                "Labels file not found"
            )

            return False

        RECOGNIZER = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        RECOGNIZER.read(
            trainer_path
        )

        LABELS = np.load(
            labels_path,
            allow_pickle=True
        ).item()

        FACE_DETECTOR = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades
                +
                "haarcascade_frontalface_default.xml"
            )
        )

        if FACE_DETECTOR.empty():

            print(
                "Face detector failed"
            )

            return False

        SYSTEM_READY = True

        print(
            "Face recognition model loaded"
        )

        return True

    except Exception as e:

        print(
            "Face system initialization error:",
            e
        )

        return False


# =========================================================
# FACE RECOGNITION
# =========================================================

def recognize_face(timeout=6):

    if not init_face_system():

        return "unknown"

    if not init_camera():

        return "unknown"

    start_time = time.time()

    detected_name = "unknown"

    consecutive_matches = 0

    last_name = None

    REQUIRED_MATCHES = 3

    CONFIDENCE_THRESHOLD = 65

    while True:

        if (
            time.time()
            - start_time
            > timeout
        ):

            break

        ret, frame = CAMERA.read()

        if not ret or frame is None:

            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = FACE_DETECTOR.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=6,

            minSize=(80, 80)

        )

        if len(faces) == 0:

            consecutive_matches = 0

            continue

        x, y, w, h = max(

            faces,

            key=lambda face:
            face[2] * face[3]

        )

        face_img = gray[
            y:y+h,
            x:x+w
        ]

        face_img = cv2.resize(

            face_img,

            (200, 200)

        )

        face_img = cv2.equalizeHist(
            face_img
        )

        label, confidence = (
            RECOGNIZER.predict(
                face_img
            )
        )

        if confidence < CONFIDENCE_THRESHOLD:

            name = LABELS.get(
                label,
                "unknown"
            )

            if name == last_name:

                consecutive_matches += 1

            else:

                last_name = name

                consecutive_matches = 1

            if (
                consecutive_matches
                >= REQUIRED_MATCHES
            ):

                detected_name = name

                print(
                    f"Face recognized: "
                    f"{name}"
                )

                break

        else:

            consecutive_matches = 0

            last_name = None

    return detected_name


# =========================================================
# CLEANUP
# =========================================================

def cleanup_camera():

    global CAMERA

    if CAMERA is not None:

        CAMERA.release()

        CAMERA = None

    cv2.destroyAllWindows()