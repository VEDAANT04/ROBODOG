
import cv2
import numpy as np

# HSV ranges for common colours
COLOR_RANGES = {
    "RED": [
        (np.array([0, 100, 80]), np.array([10, 255, 255])),
        (np.array([170, 100, 80]), np.array([179, 255, 255]))
    ],
    "GREEN": [
        (np.array([35, 70, 70]), np.array([85, 255, 255]))
    ],
    "BLUE": [
        (np.array([90, 70, 70]), np.array([130, 255, 255]))
    ]
}

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera not detected!")

print("RoboDog Colour Detection Started!")
print("Press Q to quit.")

kernel = np.ones((5, 5), np.uint8)

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame")
            break

        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(
            frame, cv2.COLOR_BGR2HSV
        )

        for color_name, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # Remove small amounts of noise
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel
            )

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    color_name,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Draw centre of detected object
                center_x = x + w // 2
                center_y = y + h // 2

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 0),
                    -1
                )

        cv2.imshow(
            "RoboDog - Colour Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
