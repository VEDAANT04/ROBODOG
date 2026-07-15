import subprocess
import sys
import time

print("=" * 60)
print("🤖 ROBODOG E2E FACE RECOGNITION WORKFLOW")
print("=" * 60)

# Step 1: Capture faces
print("\n[STEP 1/3] 📸 Capturing faces...")
try:
    # Run capturefaces.py. It will prompt for name and auto-capture 50 images.
    capture_proc = subprocess.run([sys.executable, "capturefaces.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Capture process failed or cancelled: {e}")
    sys.exit(1)

# Step 2: Train faces
print("\n[STEP 2/3] 🧠 Training model...")
try:
    # Run trainfaces.py. It will train the model using the dataset folder.
    train_proc = subprocess.run([sys.executable, "trainfaces.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training process failed: {e}")
    sys.exit(1)

# Step 3: Run Face Recognition & Generate Output
print("\n[STEP 3/3] 🔍 Verifying Face Recognition & Saving Output Image...")
try:
    from facerecognizer import recognize_face, init_camera, cleanup_camera
    import atexit
    
    # Register cleanup
    atexit.register(cleanup_camera)
    
    # Warm up camera
    print("📹 Initializing camera for verification...")
    if init_camera():
        print("👀 Look at the camera to verify your face...")
        # Run recognition with save_output=True so it writes the recognized image inside dataset/{name}/
        name = recognize_face(save_output=True)
        print(f"\n🎯 Recognition result: Recognized as '{name}'")
        if name != "unknown":
            print(f"✅ Success! The output image is saved in 'dataset/{name}/{name}_recognized.jpg'.")
        else:
            print("⚠️ Face was not recognized within the timeout period or unrecognized.")
    else:
        print("❌ Cannot initialize camera for verification.")
        
except Exception as e:
    print(f"❌ Verification failed: {e}")
finally:
    cleanup_camera()

print("\n" + "=" * 60)
print("🎉 WORKFLOW COMPLETE!")
print("=" * 60)
