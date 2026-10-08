
import cv2
from ultralytics import YOLO

# Load a small pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera not detected!")
    exit()

print("RoboDog AI Vision Started!")
print("Press Q to stop.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        # Detect objects in the frame
        results = model.predict(
            frame,
            conf=0.5,
            verbose=False
        )

        # Draw boxes around detected objects
        annotated_frame = results[0].plot()

        cv2.imshow("RoboDog - AI Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
