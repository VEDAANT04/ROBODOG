import cv2
import os
import time
import numpy as np

print("=" * 50)
print("📸 FACE CAPTURE UTILITY")
print("=" * 50)

cam = cv2.VideoCapture(0)

# Set camera properties
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)

if not cam.isOpened():
    print("❌ ERROR: Cannot open camera!")
    exit(1)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

name = input("\n👤 Enter person's name: ").lower().strip()

if not name:
    print("❌ Name cannot be empty!")
    exit(1)

path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)
    print(f"✅ Created folder: {path}")
else:
    existing = len(os.listdir(path))
    print(f"⚠️  Folder already exists with {existing} images")
    response = input("   Overwrite? (y/n): ").lower()
    if response != 'y':
        exit(1)

print("\n" + "=" * 50)
print("📸 CAPTURING FACES")
print("=" * 50)
print("✓ Look at camera from different angles")
print("✓ Move closer and farther from camera")
print("✓ Change lighting/expressions")
print("✓ Press ESC to stop early")
print("=" * 50 + "\n")

count = 0
no_face_frames = 0
start_time = time.time()
frame_skip_counter = 0  # Frame skipping for faster diverse captures
last_capture_time = 0   # Rate limit auto-captures for diversity (250ms interval)
capture_interval = 0.25

try:
    # Target 50 images per person
    while count < 50:
        ret, img = cam.read()
        
        if not ret or img is None:
            print("❌ Camera read error")
            break
        
        # Optimize frame processing by skipping alternate frames
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:  # Process every 2nd frame for speed
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face detection with optimized parameters
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Increased for stability
            minNeighbors=6,       # Increased for less sensitive detection
            minSize=(80, 80)      # Increased for better quality
        )
        
        # Draw all detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Updated progress bar
        progress = int((count / 50) * 50)
        progress_bar = "█" * progress + "░" * (50 - progress)
        
        # Display info
        info_text = f"Images: {count}/50 | Faces detected: {len(faces)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(img, progress_bar, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
        
        cv2.imshow('📸 Capture Faces', img)
        
        # Reduced from 100ms to 50ms for faster response
        key = cv2.waitKey(50) & 0xFF
        
        if key == 27:  # ESC
            print("\n⏹️  Capture cancelled by user")
            break
        elif key == 32 or (len(faces) > 0 and (time.time() - last_capture_time) >= capture_interval):  # Space or auto-capture if face detected and interval elapsed

            if len(faces) > 0:
                # Save largest face
                best_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = best_face
                face_img = gray[y:y+h, x:x+w]
                
                # Image quality enhancement for better training
                # Apply histogram equalization for better contrast
                face_img = cv2.equalizeHist(face_img)
                
                # Check brightness to avoid too dark/bright images
                brightness = np.mean(face_img)
                if brightness < 30 or brightness > 225:  # Reject too dark/bright images
                    print(f"⚠️  Frame too dark/bright (brightness: {int(brightness)}) - try adjusting lighting")
                    continue
                
                # Resize to standard size
                face_img = cv2.resize(face_img, (200, 200))
                
                count += 1
                filename = f"{path}/{count:03d}.jpg"  # Format for 50+ images
                cv2.imwrite(filename, face_img)
                print(f"✅ Captured {count}/50 - {filename}")
                last_capture_time = time.time()
                no_face_frames = 0
            else:
                print("⚠️  No face detected - position yourself in front of camera")
        else:
            if len(faces) == 0:
                no_face_frames += 1
                if no_face_frames % 15 == 0:  # Reduced frequency of messages
                    print("⚠️  No face detected - try adjusting lighting or position")
    
    print("\n" + "=" * 50)
    if count >= 50:
        elapsed_time = time.time() - start_time
        print(f"✅ Successfully captured {count} images!")
        print(f"⏱️  Time taken: {int(elapsed_time)} seconds")
    else:
        print(f"✅ Captured {count} images (target was 50)")
    print("=" * 50)
    print("\n📌 Next step: Run trainfaces.py to train the model")

finally:
    cam.release()
    cv2.destroyAllWindows()