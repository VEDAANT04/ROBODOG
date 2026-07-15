import cv2
import os
import numpy as np

dataset_path = "dataset"

print("🚀 Starting face training...")
print("=" * 60)

# Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: '{dataset_path}' folder not found!")
    print("   → Create a 'dataset' folder with subfolders for each person")
    print("   → Structure: dataset/person_name/image1.jpg")
    print("   → Run capturefaces.py first to capture images")
    exit(1)

# Initialize
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except Exception as e:
    print(f"❌ ERROR: Could not create recognizer: {e}")
    print("   → Make sure opencv-contrib-python is installed")
    print("   → pip install opencv-contrib-python")
    exit(1)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_detector.empty():
    print("❌ ERROR: Could not load face detector!")
    exit(1)

faces = []
labels = []
label_map = {}
current_id = 0
total_images = 0
trained_images = 0

# Load training data
people = os.listdir(dataset_path)
people = [p for p in people if os.path.isdir(os.path.join(dataset_path, p))]

print(f"📁 Found {len(people)} person(s) in dataset\n")

if len(people) == 0:
    print("❌ ERROR: No person folders found in dataset!")
    print("   → Create folders: dataset/person_name/")
    exit(1)

for person_name in people:
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    print(f"👤 Processing: {person_name}")
    label_map[current_id] = person_name
    
    images_in_folder = os.listdir(person_path)
    
    if len(images_in_folder) == 0:
        print(f"   ⚠️  No images found in folder!")
        current_id += 1
        continue
    
    successful = 0
    
    for image_name in images_in_folder:
        # Skip non-image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        img_path = os.path.join(person_path, image_name)
        
        try:
            # Read image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"   ⚠️  Could not read: {image_name}")
                continue
            
            total_images += 1
            
            # Detect faces in image
            faces_detected = face_detector.detectMultiScale(
                img,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(50, 50)
            )
            
            if len(faces_detected) == 0:
                print(f"   ⚠️  No face found in: {image_name}")
                continue
            
            # Use largest face if multiple detected
            best_face = max(faces_detected, key=lambda f: f[2] * f[3])
            x, y, w, h = best_face
            
            # Crop and store face
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))  # Normalize size
            
            faces.append(face_img)
            labels.append(current_id)
            successful += 1
            trained_images += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {image_name}: {e}")
            continue
    
    print(f"   ✅ {successful}/{len(images_in_folder)} images trained\n")
    current_id += 1

# Check if we have enough training data
if len(faces) < 2:
    print("❌ ERROR: Not enough training images!")
    print("   → Need at least 10 images per person")
    print(f"   → Currently have: {trained_images} trained images")
    print("   → Run capturefaces.py to capture more images")
    exit(1)

print("=" * 60)
print(f"\n🧠 Training with {trained_images} images from {len(label_map)} people...")
print(f"   People: {', '.join(label_map.values())}\n")

try:
    # Train recognizer
    recognizer.train(faces, np.array(labels))
    
    # Create models directory
    if not os.path.exists("models"):
        os.makedirs("models")
        print("✅ Created models/ directory")
    
    # Save model and labels
    recognizer.save("models/trainer.yml")
    np.save("models/labels.npy", label_map)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"   → Model saved to: models/trainer.yml")
    print(f"   → Labels saved to: models/labels.npy")
    print(f"   → Recognized people: {', '.join(label_map.values())}")
    print("\n📌 Next step: Run main.py to start the robodog!")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()