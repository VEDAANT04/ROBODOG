import cv2
import os
import numpy as np


DATASET_PATH = "dataset"

MODEL_PATH = "models/trainer.yml"

LABELS_PATH = "models/labels.npy"

IMAGE_SIZE = 200


os.makedirs(

    "models",

    exist_ok=True

)


recognizer = (

    cv2.face
    .LBPHFaceRecognizer_create()

)

faces = []

labels = []

label_map = {}

current_label = 0


# =========================================================
# LOAD DATASET
# =========================================================

for person_name in sorted(

    os.listdir(DATASET_PATH)

):

    person_path = os.path.join(

        DATASET_PATH,

        person_name

    )

    if not os.path.isdir(

        person_path

    ):

        continue

    label_map[
        current_label
    ] = person_name

    image_count = 0

    for filename in os.listdir(

        person_path

    ):

        if not filename.lower().endswith(

            (".jpg", ".jpeg", ".png")

        ):

            continue

        image_path = os.path.join(

            person_path,

            filename

        )

        image = cv2.imread(

            image_path,

            cv2.IMREAD_GRAYSCALE

        )

        if image is None:

            continue

        image = cv2.resize(

            image,

            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )

        )

        image = cv2.equalizeHist(
            image
        )

        faces.append(
            image
        )

        labels.append(
            current_label
        )

        image_count += 1

    print(

        f"{person_name}: "
        f"{image_count} images"

    )

    current_label += 1


# =========================================================
# TRAIN
# =========================================================

if len(faces) == 0:

    print(
        "ERROR: No training images found"
    )

    exit()


print()

print(
    "Training face recognition model..."
)

recognizer.train(

    faces,

    np.array(

        labels

    )

)


# =========================================================
# SAVE MODEL
# =========================================================

recognizer.write(

    MODEL_PATH

)

np.save(

    LABELS_PATH,

    label_map

)


print()

print(
    "TRAINING COMPLETE"
)

print(
    f"People trained: "
    f"{len(label_map)}"
)

print(
    f"Total images: "
    f"{len(faces)}"
)

print(
    f"Model saved: "
    f"{MODEL_PATH}"
)