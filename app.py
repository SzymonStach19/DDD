"""
Real-time detection and main application for drowsiness detection
"""
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from analyzer import ImageDrowsinessAnalyzer
from gui import DrowsinessAnalyzerGUI
from models import train_model_resnet50v2, train_model_cnn

load_model = tf.keras.models.load_model

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
        # Bardzo duży margines, by objąć całą dolną część twarzy
        margin_h = int(h * 0.40)
        margin_w = int(w * 0.25)
        y1 = max(0, 0 - margin_h)
        y2 = min(h, h + margin_h)
        x1 = max(0, 0 - margin_w)
        x2 = min(w, w + margin_w)
        enlarged_roi = face_roi[y1:y2, x1:x2]
        processed_face = self.preprocess_for_prediction(enlarged_roi, 'face')
        predictions = self.model.predict(processed_face, verbose=0)
        yawn_pred = predictions[0][3]
        no_yawn_pred = predictions[0][2]
        # Jeszcze niższy próg i mniejszy margines przewagi
        if (yawn_pred > 0.45) and (yawn_pred - no_yawn_pred > 0.05):
            return 3, yawn_pred  # 3 = 'yawn'
        return 2, no_yawn_pred  # 2 = 'no_yawn'
    
    def draw_eye_contour(self, image, landmarks, indices, color):
        points = landmarks[indices]
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=1)  # zawsze cienkie

    def draw_mouth_contour(self, image, landmarks, color):
        mouth_indices = list(range(61, 80))
        points = landmarks[mouth_indices]
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    
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
            
            # Inicjalizacja zmiennych dla etykiet
            yawn_label, eye_label, state_label = None, None, None
            eye_state = 1  # Default open
            yawn_state = 2  # Default no yawning
            eye_confidence = 0
            yawn_confidence = 0
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
                    
                    # Yawn detection (if enabled)
                    if self.detection_mode in ['yawn', 'both']:
                        face_roi = frame[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:
                            yawn_state, yawn_confidence = self.predict_yawn_state(face_roi)
                            yawn_label = f"Yawn: {self.classes[yawn_state]} ({yawn_confidence:.2f})"
                    # Eye detection (if enabled)
                    if self.detection_mode in ['eyes', 'both']:
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
                            eye_roi = frame[left_y_min:left_y_max, left_x_min:left_x_max]
                            if eye_roi.size > 0:
                                eye_state, eye_confidence = self.predict_eye_state(eye_roi)
                                # Obramówka oka zawsze zielona
                                self.draw_eye_contour(frame, landmarks, left_eye_indices, (0, 255, 0))
                                eye_label = f"Eye: {self.classes[eye_state]} ({eye_confidence:.2f})"
                        except:
                            pass
                    # State label
                    current_state = self.get_current_state(eye_state, yawn_state)
                    state_label = f"State: {current_state}"
            
            # --- WYŚWIETLANIE WSZYSTKICH NAPISÓW W JEDNYM MIEJSCU ---
            y_offset = 40
            x_offset = 10
            # Kolory napisów
            eye_text_color = (0,255,0) if eye_state == 1 else (0,0,255)
            yawn_text_color = (0,255,0) if yawn_state == 2 else (0,0,255)
            if state_label:
                cv2.putText(frame, state_label, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if eye_label and self.detection_mode in ['eyes', 'both']:
                cv2.putText(frame, eye_label, (x_offset, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_text_color, 1)
            if yawn_label and self.detection_mode in ['yawn', 'both']:
                cv2.putText(frame, yawn_label, (x_offset, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_text_color, 1)
            # --- tryb info ---
            mode_text = f"Mode: {mode_names[self.detection_mode]}"
            cv2.putText(frame, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Wyświetl obraz
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
        model_path = "models/drowsiness_resnet50v2.h5"
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
        model_path = "models/drowsiness_cnn.h5"
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
