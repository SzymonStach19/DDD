"""
Utility functions for loading and processing datasets for drowsiness detection
"""
import os
import cv2
import numpy as np

def load_dataset_resnet(data_path):
    """Loads dataset from Closed, Open, yawn, no_yawn folders for ResNet50V2 (224x224, 3 channels)."""
    images = []
    labels = []
    label_mapping = {'Closed': 0, 'Open': 1, 'no_yawn': 2, 'yawn': 3}
    for folder_name, label_index in label_mapping.items():
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist!")
            continue
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                if img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                images.append(img)
                labels.append(label_index)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return np.array(images), np.array(labels)

def load_dataset_cnn(data_path, img_size=(150, 150)):
    images, labels = [], []
    class_names = ['Closed', 'Open', 'no_yawn', 'yawn']
    for idx, label in enumerate(class_names):
        folder = os.path.join(data_path, label)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                img = cv2.imread(fpath)
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
                labels.append(idx)
            except Exception:
                images.append(np.zeros(img_size, dtype=np.uint8))
                labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
