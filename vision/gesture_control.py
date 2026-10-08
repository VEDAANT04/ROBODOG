
import cv2
import time
import mediapipe as mp
from pathlib import Path

# Locate the MediaPipe model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "gesture_recognizer.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Map hand gestures to RoboDog commands
GESTURE_COMMANDS = {
    "Open_Palm": "STOP",
    "Closed_Fist": "SIT",
    "Pointing_Up": "STAND"
}

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

print("RoboDog Gesture Recognition Started!")
print("Press Q to quit.")

start_time = time.monotonic()
last_timestamp = -1

try:
    with GestureRecognizer.create_from_options(options) as recognizer:

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms, last_timestamp + 1
            )
            last_timestamp = timestamp_ms

            result = recognizer.recognize_for_video(
                mp_image, timestamp_ms
            )

            command = "NO GESTURE"
            detected_name = "None"
            confidence = 0.0

            if result.gestures:
                gesture = result.gestures[0][0]

                detected_name = gesture.category_name
                confidence = gesture.score

                if confidence >= 0.40:
                    command = GESTURE_COMMANDS.get(
                        detected_name,
                        "UNKNOWN"
                    )

            # Display raw detection and confidence
            cv2.putText(
                frame,
                f"Detected: {detected_name} ({confidence:.2f})",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display RoboDog command
            cv2.putText(
                frame,
                f"RoboDog Command: {command}",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "RoboDog - Gesture Control",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
