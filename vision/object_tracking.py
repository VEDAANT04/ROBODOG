
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

print("RoboDog Object Tracking Started!")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.3,
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow(
            "RoboDog - Object Tracking",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
