import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

DEFAULT_IMAGE_SIZE = (256, 256)
root_dir = './PlantVillage'
train_dir = os.path.join(root_dir, 'train') if os.path.exists(os.path.join(root_dir, 'train')) else root_dir

image_list, label_list = [], []

print("[INFO] Loading sample images from dataset...")
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path)[:30]: # Fast training
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, DEFAULT_IMAGE_SIZE)
                    image_list.append(img_to_array(img))
                    label_list.append(folder)

x_data = np.array(image_list, dtype=np.float16) / 255.0
lb = LabelBinarizer()
y_data = lb.fit_transform(label_list)

# Save labels
with open('plant_disease_label_transform.pkl', 'wb') as f:
    pickle.dump(lb, f)

# Simple CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("[INFO] Training model...")
model.fit(x_data, y_data, epochs=3, batch_size=16)

# Save model
with open('plant_disease_classification_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("[SUCCESS] plant_disease_classification_model.pkl generated successfully!")