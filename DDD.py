# -*- coding: utf-8 -*-

# Import libraries
import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
from sklearn.model_selection import train_test_split

# Use tf.keras.* for all Keras components
load_model = tf.keras.models.load_model
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
ResNet50V2 = tf.keras.applications.ResNet50V2
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
Adam = tf.keras.optimizers.Adam

# --- DATASET LOADING FOR RESNET50V2 ---
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

# --- RESNET50V2 TRAINING PIPELINE ---
def train_model_resnet50v2(data_path, epochs=30, save_path='drowsiness_resnet50v2.h5'):
    """Trains a ResNet50V2-based model with strong augmentation, class weighting, and two-stage fine-tuning."""
    img_size = (224, 224)
    batch_size = 32
    num_classes = 4
    class_names = ['Closed', 'Open', 'no_yawn', 'yawn']

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))

    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=img_size + (3,))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n🔁 Training top layers...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    for layer in base_model.layers[-50:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n🔁 Fine-tuning last 50 ResNet layers...")
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
        ]
    )

    model.save(save_path)
    print(f"Model saved as {save_path}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'] + history_ft.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'] + history_ft.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'] + history_ft.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'] + history_ft.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics_optimized.png')
    plt.close()

    val_gen.reset()
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_optimized.png")
    plt.close()
    return model, history, history_ft

# --- CNN TRAINING PIPELINE ---
def train_model_cnn(data_path, epochs=50, save_path='drowsiness_cnn.h5'):
    img_size = (150, 150)
    images, labels = load_dataset_cnn(data_path, img_size)
    images = np.expand_dims(images, axis=-1)
    x_train, x_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, shuffle=True, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)
    def random_contrast(image, lower=0.9, upper=1.1):
        return tf.image.random_contrast(image, lower=lower, upper=upper)
    def custom_preprocess(image):
        image = random_contrast(image)
        return image
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        preprocessing_function=custom_preprocess
    )
    val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
    val_generator = val_test_datagen.flow(x_val, y_val, batch_size=32)
    test_generator = val_test_datagen.flow(x_test, y_test, batch_size=32)
    input_shape = images.shape[1:]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    model.save(save_path)
    print(f"Model saved as {save_path}")
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history_cnn.png')
    plt.close()
    # Evaluate and plot confusion matrix
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Closed', 'Open', 'no_yawn', 'yawn'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_cnn.png')
    plt.close()
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Closed', 'Open', 'no_yawn', 'yawn']))
    return model, history

# Class for analyzing individual photos with detection mode selection
class ImageDrowsinessAnalyzer:
    def __init__(self, model_path=None, model=None):
        """Initializes drowsiness analyzer for individual photos"""
        self.model = model if model is not None else load_model(model_path)
        # Detect model type by input shape
        input_shape = self.model.input_shape
        if len(input_shape) == 4 and input_shape[1:4] == (224, 224, 3):
            self.is_transfer = True  # ResNet50V2
        else:
            self.is_transfer = False  # Classic CNN
        
        # Class mapping
        self.classes = {
            0: "Closed eyes",
            1: "Open eyes", 
            2: "No yawning",
            3: "Yawning"
        }
        
        # Colors for different states
        self.colors = {
            0: (0, 0, 255),    # Red for closed eyes
            1: (0, 255, 0),    # Green for open eyes
            2: (0, 255, 0),    # Green for no yawning
            3: (0, 165, 255)   # Orange for yawning
        }
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create custom drawing spec with thinner lines
        self.custom_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=1,        # Thinner lines (default is 2)
            circle_radius=1     # Smaller circles (default is 2)
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def preprocess_for_prediction(self, roi, roi_type='face'):
        """Processes ROI to format required by the selected model type"""
        if self.is_transfer:
            # ResNet50V2: color, 224x224
            img = cv2.resize(roi, (224, 224))
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
        else:
            # Classic CNN: grayscale, 150x150
            img = cv2.resize(roi, (150, 150))
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
        return img
    
    def predict_eye_state(self, eye_roi):
        """Predicts eye state (open/closed)"""
        processed_eye = self.preprocess_for_prediction(eye_roi, 'eye')
        predictions = self.model.predict(processed_eye, verbose=0)
        
        # Take only predictions for eyes (classes 0 and 1)
        eye_predictions = predictions[0][:2]
        eye_predictions = eye_predictions / np.sum(eye_predictions)  # Renormalization
        
        predicted_class = np.argmax(eye_predictions)
        confidence = np.max(eye_predictions)
        
        return predicted_class, confidence
    
    def predict_yawn_state(self, face_roi):
        """Predicts yawn state (yawning/no yawning) with improved threshold and margin logic."""
        h, w = face_roi.shape[:2]
        margin_h = int(h * 0.1)
        margin_w = int(w * 0.1)
        y1 = max(0, 0 - margin_h)
        y2 = min(h, h + margin_h)
        x1 = max(0, 0 - margin_w)
        x2 = min(w, w + margin_w)
        enlarged_roi = face_roi[y1:y2, x1:x2]
        processed_face = self.preprocess_for_prediction(enlarged_roi, 'face')
        predictions = self.model.predict(processed_face, verbose=0)
        yawn_pred = predictions[0][3]
        no_yawn_pred = predictions[0][2]
        # Nowa logika: jeśli yawn < 0.8 lub różnica < 0.15, wybierz no_yawn
        if (yawn_pred < 0.8) or (yawn_pred - no_yawn_pred < 0.15):
            return 2, no_yawn_pred  # 2 = 'no_yawn'
        return 3, yawn_pred  # 3 = 'yawn'
    
    def draw_eye_contour(self, image, landmarks, indices, color):
        points = landmarks[indices]
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

    def draw_mouth_contour(self, image, landmarks, color):
        mouth_indices = list(range(61, 80))
        points = landmarks[mouth_indices]
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

    def analyze_image(self, image_path, detection_mode='both'):
        """
        Analyzes single image for drowsiness
        detection_mode: 'eyes' - only eyes, 'yawn' - only yawning, 'both' - both
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Image could not be loaded."
        
        # Copy image for drawing
        result_image = image.copy()
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Face detection with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return result_image, []
        
        analysis_results = []
        
        h, w, _ = image.shape
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            face_analysis = {
                'face_id': i + 1,
                'eye_state': None,
                'yawn_state': None,
                'drowsiness_level': 'Unknown',
                'confidence_eye': 0,
                'confidence_yawn': 0,
                'detection_mode': detection_mode
            }
            
            # Draw face mesh with thinner lines
            self.mp_drawing.draw_landmarks(
                result_image,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                None,
                self.custom_drawing_spec  # Use custom thin drawing spec
            )
            
            # Get face landmark coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append([x, y])
            landmarks = np.array(landmarks)

            # Calculate bounding box for entire face
            x_min, y_min = np.min(landmarks, axis=0)
            x_max, y_max = np.max(landmarks, axis=0)

            eye_state, yawn_state = None, None
            # Yawn detection
            if detection_mode in ['yawn', 'both']:
                face_roi = image[y_min:y_max, x_min:x_max]
                if face_roi.size > 0:
                    yawn_state, conf_yawn = self.predict_yawn_state(face_roi)
                    face_analysis['yawn_state'] = self.classes[yawn_state]
                    face_analysis['confidence_yawn'] = conf_yawn
            # Eye detection
            if detection_mode in ['eyes', 'both']:
                left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                try:
                    left_eye_points = landmarks[left_eye_indices]
                    left_x_min, left_y_min = np.min(left_eye_points, axis=0)
                    left_x_max, left_y_max = np.max(left_eye_points, axis=0)
                    margin = 10
                    left_x_min = max(0, left_x_min - margin)
                    left_y_min = max(0, left_y_min - margin)
                    left_x_max = min(w, left_x_max + margin)
                    left_y_max = min(h, left_y_max + margin)
                    eye_roi = image[left_y_min:left_y_max, left_x_min:left_x_max]
                    if eye_roi.size > 0:
                        eye_state, conf_eye = self.predict_eye_state(eye_roi)
                        face_analysis['eye_state'] = self.classes[eye_state]
                        face_analysis['confidence_eye'] = conf_eye
                except Exception as e:
                    pass
            
            # Simplified drowsiness level assessment
            drowsiness_level = self.assess_simple_drowsiness(eye_state, yawn_state, detection_mode)
            face_analysis['drowsiness_level'] = drowsiness_level
            
            # Label for drowsiness level
            drowsiness_label = f"Face {i+1} - State: {drowsiness_level}"
            cv2.putText(result_image, drowsiness_label, (x_min, y_min-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            analysis_results.append(face_analysis)
        
        return result_image, analysis_results
    
    def assess_simple_drowsiness(self, eye_state, yawn_state, detection_mode):
        """Simplified drowsiness level assessment without time counters"""
        if detection_mode == 'eyes':
            # Only eye analysis
            if eye_state == 0:  # Closed eyes
                return "Closed eyes (Drowsy)"
            else:
                return "Open eyes (Alert)"
        
        elif detection_mode == 'yawn':
            # Only yawn analysis
            if yawn_state == 3:  # Yawning
                return "Yawning (Drowsy)"
            else:
                return "No yawning (Alert)"
        
        else:  # detection_mode == 'both'
            # Combined analysis
            if eye_state == 0 and yawn_state == 3:  # Closed eyes + yawning
                return "Closed eyes + Yawning (Very Drowsy)"
            elif eye_state == 0:  # Only closed eyes
                return "Closed eyes (Drowsy)"
            elif yawn_state == 3:  # Only yawning
                return "Yawning (Drowsy)"
            else:  # Open eyes, no yawning
                return "Alert"

# GUI class with detection mode selection
class DrowsinessAnalyzerGUI:
    def __init__(self, analyzer):
        """Initializes graphical interface with detection mode options"""
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Drowsiness Analyzer - Photo Analysis")
        self.root.geometry("900x700")
        
        # Variable for detection mode
        self.detection_mode = tk.StringVar(value='both')
        
        self.setup_gui()
    
    def setup_gui(self):
        """Configures graphical interface"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame for detection mode options
        mode_frame = tk.LabelFrame(main_frame, text="Detection Mode", font=("Arial", 12))
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Radio buttons for mode selection
        modes = [
            ('Eyes only', 'eyes'),
            ('Yawning only', 'yawn'),
            ('Eyes and yawning', 'both')
        ]
        
        for text, value in modes:
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.detection_mode, 
                               value=value, font=("Arial", 10))
            rb.pack(side=tk.LEFT, padx=20, pady=10)
        
        # File selection button
        select_button = tk.Button(main_frame, text="Select Photo", 
                                 command=self.select_image, 
                                 font=("Arial", 12), 
                                 bg="#4CAF50", fg="white",
                                 height=2, width=20)
        select_button.pack(pady=10)
        
        # Frame for image
        self.image_frame = tk.Frame(main_frame, bg="lightgray", height=400)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Label for image
        self.image_label = tk.Label(self.image_frame, text="Select photo for analysis", 
                                   font=("Arial", 14), bg="lightgray")
        self.image_label.pack(expand=True)
        
        # Frame for results
        self.results_frame = tk.Frame(main_frame, height=100)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Text widget for results
        self.results_text = tk.Text(self.results_frame, font=("Arial", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def select_image(self):
        """Opens file selection dialog and analyzes image"""
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.analyze_selected_image(file_path)
    
    def analyze_selected_image(self, image_path):
        """Analyzes selected image with chosen detection mode"""
        try:
            # Get selected detection mode
            mode = self.detection_mode.get()
            
            # Image analysis
            result_image, analysis_results = self.analyzer.analyze_image(image_path, mode)
            
            if isinstance(analysis_results, str):  # Error
                messagebox.showerror("Error", analysis_results)
                return
            
            # Display result image
            self.display_image(result_image)
            
            # Display analysis results
            self.display_results(analysis_results, image_path, mode)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
    
    def display_image(self, cv_image):
        """Displays image in GUI"""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Resize image for display
        height, width = rgb_image.shape[:2]
        max_height = 400
        max_width = 600
        
        if height > max_height or width > max_width:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display in label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep reference
    
    def display_results(self, analysis_results, image_path, detection_mode):
        """Displays analysis results with mode information"""
        self.results_text.delete(1.0, tk.END)
        
        mode_names = {
            'eyes': 'EYES ONLY',
            'yawn': 'YAWNING ONLY', 
            'both': 'EYES AND YAWNING'
        }
        
        results_text = f"IMAGE ANALYSIS: {os.path.basename(image_path)}\n"
        results_text += f"DETECTION MODE: {mode_names[detection_mode]}\n"
        results_text += "=" * 50 + "\n\n"
        
        if not analysis_results:
            results_text += "No faces detected in image.\n"
        else:
            for result in analysis_results:
                results_text += f"FACE {result['face_id']}:\n"
                if detection_mode in ['eyes', 'both']:
                    results_text += f"  Eye state: {result['eye_state']}"
                    if result['confidence_eye'] > 0:
                        results_text += f" (confidence: {result['confidence_eye']:.2%})"
                    results_text += "\n"
                if detection_mode in ['yawn', 'both']:
                    results_text += f"  Yawn state: {result['yawn_state']}"
                    results_text += f" (confidence: {result['confidence_yawn']:.2%})\n"
                results_text += f"  State: {result['drowsiness_level']}\n"
                results_text += "-" * 30 + "\n"
        self.results_text.insert(1.0, results_text)
    
    def run(self):
        """Runs GUI"""
        self.root.mainloop()

# Real-time detection class with face mesh
class RealTimeDrowsinessDetector:
    def __init__(self, model_path=None, model=None, detection_mode='both'):
        """Initializes real-time drowsiness detector with face mesh"""
        self.model = model if model is not None else load_model(model_path)
        # Detect model type by input shape
        input_shape = self.model.input_shape
        if len(input_shape) == 4 and input_shape[1:4] == (224, 224, 3):
            self.is_transfer = True  # ResNet50V2
        else:
            self.is_transfer = False  # Classic CNN
        self.detection_mode = detection_mode
        
        # Class mapping
        self.classes = {
            0: "Closed eyes",
            1: "Open eyes", 
            2: "No yawning",
            3: "Yawning"
        }
        
        # Colors for different states
        self.colors = {
            0: (0, 0, 255),    # Red for closed eyes
            1: (0, 255, 0),    # Green for open eyes
            2: (0, 255, 0),    # Green for no yawning
            3: (0, 165, 255)   # Orange for yawning
        }
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create custom drawing spec with thinner lines
        self.custom_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=1,        # Thinner lines (default is 2)
            circle_radius=1     # Smaller circles (default is 2)
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def preprocess_for_prediction(self, roi, roi_type='face'):
        """Processes ROI to format required by model type (ResNet50V2 or classic CNN)"""
        if self.is_transfer:
            img = cv2.resize(roi, (224, 224))
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.astype(np.float32) / 255.0
        else:
            img = cv2.resize(roi, (150, 150))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_eye_state(self, eye_roi):
        """Predicts eye state (open/closed)"""
        processed_eye = self.preprocess_for_prediction(eye_roi, 'eye')
        predictions = self.model.predict(processed_eye, verbose=0)
        
        # Take only predictions for eyes (classes 0 and 1)
        eye_predictions = predictions[0][:2]
        eye_predictions = eye_predictions / np.sum(eye_predictions)  # Renormalization
        
        predicted_class = np.argmax(eye_predictions)
        confidence = np.max(eye_predictions)
        
        return predicted_class, confidence
    
    def predict_yawn_state(self, face_roi):
        """Predicts yawn state (yawning/no yawning) with improved threshold and margin logic."""
        h, w = face_roi.shape[:2]
        margin_h = int(h * 0.1)
        margin_w = int(w * 0.1)
        y1 = max(0, 0 - margin_h)
        y2 = min(h, h + margin_h)
        x1 = max(0, 0 - margin_w)
        x2 = min(w, w + margin_w)
        enlarged_roi = face_roi[y1:y2, x1:x2]
        processed_face = self.preprocess_for_prediction(enlarged_roi, 'face')
        predictions = self.model.predict(processed_face, verbose=0)
        yawn_pred = predictions[0][3]
        no_yawn_pred = predictions[0][2]
        if (yawn_pred < 0.8) or (yawn_pred - no_yawn_pred < 0.15):
            return 2, no_yawn_pred
        return 3, yawn_pred
    
    def run_detection(self, camera_index=0):
        """Runs real-time drowsiness detection with face mesh"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Cannot open camera!")
            return
        
        mode_names = {
            'eyes': 'EYES ONLY',
            'yawn': 'YAWNING ONLY',
            'both': 'EYES AND YAWNING'
        }
        
        print(f"Starting real-time drowsiness detection...")
        print(f"Detection mode: {mode_names[self.detection_mode]}")
        print("Press 'q' to quit")
        print("Press '1' - eyes only, '2' - yawning only, '3' - both")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read frame from camera!")
                break
            
            # Check if mode was changed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                self.detection_mode = 'eyes'
                print("Changed mode to: EYES ONLY")
            elif key == ord('2'):
                self.detection_mode = 'yawn'
                print("Changed mode to: YAWNING ONLY")
            elif key == ord('3'):
                self.detection_mode = 'both'
                print("Changed mode to: EYES AND YAWNING")
            elif key == ord('q'):
                break
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh with thinner lines
                    self.mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        None,
                        self.custom_drawing_spec  # Use custom thin drawing spec
                    )
                    
                    # Get face landmark coordinates
                    h, w, _ = frame.shape
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append([x, y])
                    landmarks = np.array(landmarks)
                    
                    # Calculate bounding box for entire face
                    x_min, y_min = np.min(landmarks, axis=0)
                    x_max, y_max = np.max(landmarks, axis=0)
                    
                    eye_state = 1  # Default open
                    yawn_state = 2  # Default no yawning
                    
                    # Yawn detection (if enabled)
                    if self.detection_mode in ['yawn', 'both']:
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:
                            yawn_state, yawn_confidence = self.predict_yawn_state(face_roi)
                            
                            # Label for yawning
                            yawn_label = f"Yawn: {self.classes[yawn_state]} ({yawn_confidence:.2f})"
                            cv2.putText(frame, yawn_label, (x_min, y_min-30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[yawn_state], 2)
                    
                    # Eye detection (if enabled)
                    if self.detection_mode in ['eyes', 'both']:
                        # Points for left eye (MediaPipe landmarks)
                        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                        
                        try:
                            # Left eye
                            left_eye_points = landmarks[left_eye_indices]
                            left_x_min, left_y_min = np.min(left_eye_points, axis=0)
                            left_x_max, left_y_max = np.max(left_eye_points, axis=0)
                            
                            # Add margin
                            margin = 10
                            left_x_min = max(0, left_x_min - margin)
                            left_y_min = max(0, left_y_min - margin)
                            left_x_max = min(w, left_x_max + margin)
                            left_y_max = min(h, left_y_max + margin)
                            
                            eye_roi = frame[left_y_min:left_y_max, left_x_min:left_x_max]
                            
                            if eye_roi.size > 0:
                                eye_state, eye_confidence = self.predict_eye_state(eye_roi)
                                
                                # Kolor konturu oka: czerwony jeśli zamknięte, zielony jeśli otwarte
                                eye_color = (0, 0, 255) if eye_state == 0 else (0, 255, 0)
                                self.draw_eye_contour(frame, landmarks, left_eye_indices, eye_color)
                                
                                # Label for eye
                                eye_label = f"Eye: {self.classes[eye_state]} ({eye_confidence:.2f})"
                                cv2.putText(frame, eye_label, (left_x_min, left_y_min-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
                        except:
                            pass
                    
                    # Display current state
                    current_state = self.get_current_state(eye_state, yawn_state)
                    state_label = f"State: {current_state}"
                    cv2.putText(frame, state_label, (x_min, y_min-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display mode information
            mode_text = f"Mode: {mode_names[self.detection_mode]}"
            cv2.putText(frame, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Drowsiness Detection - Face Mesh', frame)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    def get_current_state(self, eye_state, yawn_state):
        """Returns current state based on detection mode"""
        if self.detection_mode == 'eyes':
            return self.classes[eye_state]
        elif self.detection_mode == 'yawn':
            return self.classes[yawn_state]
        else:  # both
            if eye_state == 0 and yawn_state == 3:
                return "Closed eyes + Yawning"
            elif eye_state == 0:
                return "Closed eyes"
            elif yawn_state == 3:
                return "Yawning"
            else:
                return "Alert"

# Main function with extended menu
def main():
    """Main program function with mode selection menu and detection options"""
    data_path = "dataset"
    model_path = None
    model = None
    print("Select model type:")
    print("1. Classic CNN")
    print("2. ResNet50V2")
    model_choice = input("Enter 1 or 2: ").strip()
    if model_choice == '2':
        model_path = "drowsiness_resnet50v2.h5"
        if os.path.exists(model_path):
            print(f"Found existing model: {model_path}")
            choice = input("Do you want to train a new model? (y/n): ").lower()
            if choice == 'y':
                model, history, history_ft = train_model_resnet50v2(data_path, epochs=30, save_path=model_path)
            else:
                print("Loading existing model...")
                model = load_model(model_path)
        else:
            print("Model not found. Starting training...")
            model, history, history_ft = train_model_resnet50v2(data_path, epochs=30, save_path=model_path)
    else:
        model_path = "drowsiness_cnn.h5"
        if os.path.exists(model_path):
            print(f"Found existing model: {model_path}")
            choice = input("Do you want to train a new model? (y/n): ").lower()
            if choice == 'y':
                model, history = train_model_cnn(data_path, epochs=50, save_path=model_path)
            else:
                print("Loading existing model...")
                model = load_model(model_path)
        else:
            print("Model not found. Starting training...")
            model, history = train_model_cnn(data_path, epochs=50, save_path=model_path)
    if model is None:
        print("Failed to load model!")
        return
    while True:
        print("\n" + "="*60)
        print("DROWSINESS ANALYZER - FACE MESH")
        print("="*60)
        print("Select working mode:")
        print("1. Single photo analysis (GUI)")
        print("2. Real-time detection (camera)")
        print("3. Command line photo analysis")
        print("4. Exit")
        print("-"*60)
        choice = input("Select option (1-4): ").strip()
        if choice == '1':
            print("Starting GUI for photo analysis...")
            analyzer = ImageDrowsinessAnalyzer(model=model)
            gui = DrowsinessAnalyzerGUI(analyzer)
            gui.run()
        elif choice == '2':
            print("\nSelect detection mode for camera:")
            print("1. Eyes only")
            print("2. Yawning only")
            print("3. Eyes and yawning")
            mode_choice = input("Select mode (1-3): ").strip()
            mode_map = {'1': 'eyes', '2': 'yawn', '3': 'both'}
            detection_mode = mode_map.get(mode_choice, 'both')
            print("Starting real-time detection...")
            detector = RealTimeDrowsinessDetector(model=model, detection_mode=detection_mode)
            detector.run_detection(camera_index=0)
        elif choice == '3':
            image_path = input("Enter photo path: ").strip()
            if os.path.exists(image_path):
                print("\nSelect detection mode:")
                print("1. Eyes only")
                print("2. Yawning only")
                print("3. Eyes and yawning")
                mode_choice = input("Select mode (1-3): ").strip()
                mode_map = {'1': 'eyes', '2': 'yawn', '3': 'both'}
                detection_mode = mode_map.get(mode_choice, 'both')
                analyzer = ImageDrowsinessAnalyzer(model=model)
                result_image, analysis_results = analyzer.analyze_image(image_path, detection_mode)
                if isinstance(analysis_results, str):
                    print(f"Error: {analysis_results}")
                else:
                    print(f"Analysis results for {os.path.basename(image_path)}:")
                    for result in analysis_results:
                        print(f"Face {result['face_id']}: {result['drowsiness_level']}")
                    save_path = os.path.splitext(image_path)[0] + '_result.jpg'
                    cv2.imwrite(save_path, result_image)
                    print(f"Result image saved as {save_path}")
            else:
                print("File does not exist!")
        elif choice == '4':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid option. Please select 1-4.")

if __name__ == "__main__":
    main()
