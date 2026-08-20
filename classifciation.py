import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
DEFAULT_IMAGE_SIZE = (256, 256)
MODEL_PATH = 'plant_disease_classification_model.pkl'
LABEL_PATH = 'plant_disease_label_transform.pkl'

# Load trained model and label binarizer
print("[INFO] Loading Model & Labels...")
model = pickle.load(open(MODEL_PATH, 'rb'))
image_labels = pickle.load(open(LABEL_PATH, 'rb'))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_disease(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    image_array = convert_image_to_array(image_path)
    if image_array.size == 0:
        print(f"[ERROR] Could not read image from {image_path}")
        return

    np_image = np.array(image_array, dtype=np.float16) / 255.0
    np_image = np.expand_dims(np_image, axis=0)

    # Prediction
    preds = model.predict(np_image)
    result = np.argmax(preds, axis=1)[0]
    predicted_label = image_labels.classes_[result]
    
    print(f"\n==========================================")
    print(f"Path: {image_path}")
    print(f"Predicted Disease: {predicted_label}")
    print(f"==========================================\n")

if __name__ == "__main__":
    # Test on local images from PlantVillage directory
    sample_dir = os.path.join(".", "PlantVillage")
    print(f"[INFO] Looking for test images inside: {sample_dir}")

    # Agar local PlantVillage folder mein files hain toh pehli available image predict karo
    found_image = False
    for root, dirs, files in os.walk(sample_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_img_path = os.path.join(root, file)
                predict_disease(test_img_path)
                found_image = True
                break
        if found_image:
            break

    if not found_image:
        print("[WARNING] Koi sample image nahi mili PlantVillage folder mein. Test image ka path manually provide karein.")