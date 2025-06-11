#!/usr/bin/env python3
"""
Driver Drowsiness Detection System
Politechnika Śląska - Computer Vision Project
Authors: Jakub Dziewior, Szymon Stach
Supervisor: mgr inż. Krzysztof Hanzel
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import pickle
import warnings

warnings.filterwarnings('ignore')


class DrowsinessDataProcessor:
    """Class for processing the Driver Drowsiness Dataset from Kaggle"""

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.img_size = (227, 227)  # As specified in project card
        self.classes = ['Non-Drowsy', 'Drowsy']

    def load_dataset(self):
        """Load and preprocess the DDD dataset"""
        print("Loading Driver Drowsiness Dataset...")

        X, y = [], []
        class_counts = {'Non-Drowsy': 0, 'Drowsy': 0}

        for class_idx, class_name in enumerate(self.classes):
            class_path = self.dataset_path / class_name

            if not class_path.exists():
                print(f"Warning: Path {class_path} not found")
                continue

            for img_path in class_path.glob('*.png'):
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0  # Normalize

                    X.append(img)
                    y.append(class_idx)
                    class_counts[class_name] += 1

                    if len(X) % 1000 == 0:
                        print(f"Processed {len(X)} images...")

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        print(f"Dataset loaded: {class_counts}")
        print(f"Total images: {len(X)}")

        return np.array(X), np.array(y)

    def augment_data(self, X, y):
        """Apply data augmentation to balance and increase dataset"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            fill_mode='nearest'
        )

        return datagen, X, y


class DrowsinessModelCNN:
    """CNN Model for drowsiness detection based on facial images"""

    def __init__(self, input_shape=(227, 227, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_model(self):
        """Build CNN architecture optimized for facial feature detection"""
        print("Building CNN model...")

        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile model - POPRAWIONA KONFIGURACJA
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        self.model = model
        print("Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")

        return model


    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the CNN model with callbacks"""
        print("Starting model training...")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_drowsiness_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        print("Training completed!")
        return self.history

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")

        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Non-Drowsy', 'Drowsy']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Drowsy', 'Drowsy'],
                    yticklabels=['Non-Drowsy', 'Drowsy'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return y_pred, y_pred_proba

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Precision
        ax3.plot(self.history.history['precision'], label='Training Precision')
        ax3.plot(self.history.history['val_precision'], label='Validation Precision')
        ax3.set_title('Model Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True)

        # Recall
        ax4.plot(self.history.history['recall'], label='Training Recall')
        ax4.plot(self.history.history['val_recall'], label='Validation Recall')
        ax4.set_title('Model Recall')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class MediaPipeAnalyzer:
    """Real-time drowsiness detection using MediaPipe"""

    def __init__(self, model_path=None):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Load trained model if provided
        self.cnn_model = None
        if model_path and os.path.exists(model_path):
            self.cnn_model = keras.models.load_model(model_path)
            print(f"CNN model loaded from {model_path}")

        # Eye landmarks (MediaPipe 468 landmarks)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Mouth landmarks for yawn detection
        self.MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        # Drowsiness tracking
        self.ear_threshold = 0.25
        self.yawn_threshold = 20
        self.consecutive_frames = 0
        self.drowsy_frames_threshold = 15

    def calculate_ear(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            # Get eye landmarks
            eye_landmarks = []
            for point in eye_points:
                x = landmarks[point].x
                y = landmarks[point].y
                eye_landmarks.append([x, y])

            eye_landmarks = np.array(eye_landmarks)

            # Calculate EAR
            # Vertical eye landmarks
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            # Horizontal eye landmark
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.25  # Default value

    def calculate_mouth_ratio(self, landmarks):
        """Calculate mouth opening ratio for yawn detection"""
        try:
            mouth_landmarks = []
            for point in self.MOUTH:
                x = landmarks[point].x
                y = landmarks[point].y
                mouth_landmarks.append([x, y])

            mouth_landmarks = np.array(mouth_landmarks)

            # Vertical mouth distance
            vertical_dist = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[9])
            # Horizontal mouth distance
            horizontal_dist = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])

            mouth_ratio = vertical_dist / horizontal_dist
            return mouth_ratio
        except:
            return 0.0

    def analyze_frame(self, frame):
        """Analyze single frame for drowsiness indicators"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        drowsiness_indicators = {
            'face_detected': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'mouth_ratio': 0.0,
            'is_drowsy': False,
            'is_yawning': False,
            'cnn_drowsy_prob': 0.0
        }

        if results.multi_face_landmarks:
            drowsiness_indicators['face_detected'] = True

            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR for both eyes
                left_ear = self.calculate_ear(face_landmarks.landmark, self.LEFT_EYE)
                right_ear = self.calculate_ear(face_landmarks.landmark, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # Calculate mouth ratio
                mouth_ratio = self.calculate_mouth_ratio(face_landmarks.landmark)

                # Update indicators
                drowsiness_indicators.update({
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear,
                    'mouth_ratio': mouth_ratio,
                    'is_yawning': mouth_ratio > self.yawn_threshold / 100.0
                })

                # Drowsiness detection based on EAR
                if avg_ear < self.ear_threshold:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 0

                drowsiness_indicators['is_drowsy'] = self.consecutive_frames >= self.drowsy_frames_threshold

                # CNN prediction if model available
                if self.cnn_model is not None:
                    try:
                        # Extract face region for CNN
                        h, w, _ = frame.shape
                        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]

                        x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                        y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

                        face_roi = frame[y_min:y_max, x_min:x_max]
                        face_roi = cv2.resize(face_roi, (227, 227))
                        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_roi = face_roi.astype(np.float32) / 255.0
                        face_roi = np.expand_dims(face_roi, axis=0)

                        cnn_pred = self.cnn_model.predict(face_roi, verbose=0)[0][0]
                        drowsiness_indicators['cnn_drowsy_prob'] = float(cnn_pred)
                    except:
                        pass

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

        return frame, drowsiness_indicators


def main():
    """Main function to demonstrate the complete system"""
    print("Driver Drowsiness Detection System")
    print("=" * 50)

    # Configuration
    DATASET_PATH = "dataset"  # Update this path
    MODEL_SAVE_PATH = "drowsiness_model.h5"

    while True:
        print("\nSelect an option:")
        print("1. Train new CNN model")
        print("2. Test real-time detection")
        print("3. Evaluate existing model")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            # Train new model
            if not os.path.exists(DATASET_PATH):
                print(f"Dataset path {DATASET_PATH} not found!")
                print("Please update DATASET_PATH variable with correct path to DDD dataset")
                continue

            # Load and process data
            processor = DrowsinessDataProcessor(DATASET_PATH)
            X, y = processor.load_dataset()

            if len(X) == 0:
                print("No data loaded. Check dataset path and structure.")
                continue

            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
                                                            stratify=y_temp)

            print(f"Training set: {len(X_train)} images")
            print(f"Validation set: {len(X_val)} images")
            print(f"Test set: {len(X_test)} images")

            # Build and train model
            model = DrowsinessModelCNN()
            model.build_model()
            model.train_model(X_train, y_train, X_val, y_val, epochs=50)

            # Evaluate model
            model.evaluate_model(X_test, y_test)
            model.plot_training_history()

            # Save model
            model.model.save(MODEL_SAVE_PATH)
            print(f"Model saved as {MODEL_SAVE_PATH}")

            # Save test data for later evaluation
            with open('test_data.pkl', 'wb') as f:
                pickle.dump((X_test, y_test), f)

        elif choice == "2":
            # Real-time detection
            analyzer = MediaPipeAnalyzer(MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else None)

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                continue

            print("Starting real-time detection. Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Analyze frame
                analyzed_frame, indicators = analyzer.analyze_frame(frame)

                # Display information
                status_color = (0, 255, 0)  # Green
                status_text = "ALERT"

                if indicators['is_drowsy'] or indicators['cnn_drowsy_prob'] > 0.7:
                    status_color = (0, 0, 255)  # Red
                    status_text = "DROWSY"
                elif indicators['is_yawning']:
                    status_color = (0, 165, 255)  # Orange
                    status_text = "YAWNING"

                # Draw status
                cv2.rectangle(analyzed_frame, (10, 10), (300, 120), (0, 0, 0), -1)
                cv2.putText(analyzed_frame, f"Status: {status_text}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(analyzed_frame, f"EAR: {indicators['avg_ear']:.3f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(analyzed_frame, f"CNN Prob: {indicators['cnn_drowsy_prob']:.3f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(analyzed_frame, f"Mouth: {indicators['mouth_ratio']:.3f}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Driver Drowsiness Detection', analyzed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif choice == "3":
            # Evaluate existing model
            if not os.path.exists(MODEL_SAVE_PATH):
                print(f"Model file {MODEL_SAVE_PATH} not found!")
                continue

            if not os.path.exists('test_data.pkl'):
                print("Test data not found! Please train the model first.")
                continue

            # Load test data
            with open('test_data.pkl', 'rb') as f:
                X_test, y_test = pickle.load(f)

            # Load and evaluate model
            model = DrowsinessModelCNN()
            model.model = keras.models.load_model(MODEL_SAVE_PATH)
            model.evaluate_model(X_test, y_test)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()