import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img

data_dir = 'preprocessed_data'
images = []
labels = []
label_map = {str(i): i for i in range(10)}  # Map for digits 0-9
label_map.update({chr(i): i - ord('A') + 10 for i in range(ord('A'), ord('Z') + 1)})  # Map for letters A-Z
label_map['_'] = 36  # Map for space

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):  # Check if it's a directory
        for img_name in os.listdir(label_dir):
            if img_name == ".DS_Store":
                continue  # Skip .DS_Store file
            img_path = os.path.join(label_dir, img_name)
            try:
                images.append(preprocess_image(img_path))
                labels.append(label_map[label])
            except ValueError as e:
                print(e)

images = np.array(images)
images = np.expand_dims(images, axis=-1)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)