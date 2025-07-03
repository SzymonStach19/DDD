"""
Image analysis class for drowsiness detection
"""
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

load_model = tf.keras.models.load_model

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
        # --- Zmienne na napisy ---
        yawn_label, eye_label, state_label = None, None, None
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
                    yawn_label = f"Yawn: {self.classes[yawn_state]} ({conf_yawn:.2f})"
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
                        eye_label = f"Eye: {self.classes[eye_state]} ({conf_eye:.2f})"
                        # Obramówka oka zawsze zielona
                        self.draw_eye_contour(result_image, landmarks, left_eye_indices, (0,255,0))
                except Exception as e:
                    pass
            # Simplified drowsiness level assessment
            drowsiness_level = self.assess_simple_drowsiness(eye_state, yawn_state, detection_mode)
            face_analysis['drowsiness_level'] = drowsiness_level
            state_label = f"State: {drowsiness_level}"
            # --- WYŚWIETLANIE WSZYSTKICH NAPISÓW W JEDNYM MIEJSCU ---
            y_offset = 40
            x_offset = 10
            eye_text_color = (0,255,0) if eye_state == 1 else (0,0,255)
            yawn_text_color = (0,255,0) if yawn_state == 2 else (0,0,255)
            if state_label:
                cv2.putText(result_image, state_label, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if eye_label and detection_mode in ['eyes', 'both']:
                cv2.putText(result_image, eye_label, (x_offset, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_text_color, 1)
            if yawn_label and detection_mode in ['yawn', 'both']:
                cv2.putText(result_image, yawn_label, (x_offset, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_text_color, 1)
            
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
