#!/usr/bin/env python3
"""
Flask API for Driver Drowsiness Detection
Serves the trained CNN model for real-time predictions
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import base64
import io
from PIL import Image
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface


class DrowsinessDetectionAPI:
    def __init__(self, model_path='drowsiness_model.h5'):
        """Initialize the API with trained model and MediaPipe"""
        self.model = None
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmarks for EAR calculation
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

        # Load model if exists
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning(f"Model file {model_path} not found")

    def preprocess_image(self, image):
        """Preprocess image for CNN prediction"""
        try:
            # Resize to 227x227 as per dataset specification
            image = cv2.resize(image, (227, 227))

            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Normalize pixel values
            image = image.astype(np.float32) / 255.0

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def calculate_ear(self, landmarks, eye_points):
        """Calculate Eye Aspect Ratio"""
        try:
            eye_landmarks = []
            for point in eye_points:
                x = landmarks[point].x
                y = landmarks[point].y
                eye_landmarks.append([x, y])

            eye_landmarks = np.array(eye_landmarks)

            # Calculate distances
            A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.25

    def calculate_mouth_ratio(self, landmarks):
        """Calculate mouth opening ratio for yawn detection"""
        try:
            mouth_points = []
            for point in self.MOUTH:
                x = landmarks[point].x
                y = landmarks[point].y
                mouth_points.append([x, y])

            mouth_points = np.array(mouth_points)

            # Calculate mouth ratio
            vertical_dist = np.linalg.norm(mouth_points[3] - mouth_points[9])
            horizontal_dist = np.linalg.norm(mouth_points[0] - mouth_points[6])

            mouth_ratio = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
            return mouth_ratio
        except:
            return 0.0

    def analyze_image(self, image):
        """Analyze image for drowsiness indicators"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        analysis_result = {
            'face_detected': False,
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'mouth_ratio': 0.0,
            'cnn_prediction': 0.0,
            'is_drowsy': False,
            'is_yawning': False,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }

        if results.multi_face_landmarks:
            analysis_result['face_detected'] = True

            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR
                left_ear = self.calculate_ear(face_landmarks.landmark, self.LEFT_EYE)
                right_ear = self.calculate_ear(face_landmarks.landmark, self.RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                # Calculate mouth ratio
                mouth_ratio = self.calculate_mouth_ratio(face_landmarks.landmark)

                analysis_result.update({
                    'left_ear': round(left_ear, 4),
                    'right_ear': round(right_ear, 4),
                    'avg_ear': round(avg_ear, 4),
                    'mouth_ratio': round(mouth_ratio, 4),
                    'is_yawning': mouth_ratio > 0.6
                })

                # CNN prediction if model is available
                if self.model is not None:
                    try:
                        # Extract face region
                        h, w, _ = image.shape
                        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
                        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]

                        x_min = max(0, min(x_coords) - 20)
                        x_max = min(w, max(x_coords) + 20)
                        y_min = max(0, min(y_coords) - 20)
                        y_max = min(h, max(y_coords) + 20)

                        face_roi = image[y_min:y_max, x_min:x_max]
                        processed_face = self.preprocess_image(face_roi)

                        if processed_face is not None:
                            cnn_pred = self.model.predict(processed_face, verbose=0)[0][0]
                            analysis_result['cnn_prediction'] = round(float(cnn_pred), 4)
                            analysis_result['confidence'] = round(float(cnn_pred), 4)
                    except Exception as e:
                        logger.error(f"CNN prediction error: {e}")

                # Determine drowsiness
                ear_drowsy = avg_ear < 0.25
                cnn_drowsy = analysis_result['cnn_prediction'] > 0.7
                analysis_result['is_drowsy'] = ear_drowsy or cnn_drowsy

                break

        return analysis_result


# Initialize the detection system
detector = DrowsinessDetectionAPI()


@app.route('/')
def index():
    """Serve the main web interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Driver Drowsiness Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .endpoint { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { color: #28a745; font-weight: bold; }
            code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
            .example { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            pre { background: #343a40; color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Driver Drowsiness Detection API</h1>

            <p><strong>Politechnika Śląska</strong> - Computer Vision Project<br>
            Authors: Jakub Dziewior, Szymon Stach</p>

            <div class="endpoint">
                <h3><span class="method">POST</span> /analyze</h3>
                <p>Analyze an image for drowsiness indicators</p>
                <p><strong>Input:</strong> Base64 encoded image in JSON format</p>
                <p><strong>Output:</strong> Drowsiness analysis results</p>

                <div class="example">
                    <strong>Example Request:</strong>
                    <pre>{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD..."
}</pre>
                </div>

                <div class="example">
                    <strong>Example Response:</strong>
                    <pre>{
    "face_detected": true,
    "left_ear": 0.2845,
    "right_ear": 0.2901,
    "avg_ear": 0.2873,
    "mouth_ratio": 0.4521,
    "cnn_prediction": 0.1234,
    "is_drowsy": false,
    "is_yawning": false,
    "confidence": 0.1234,
    "timestamp": "2024-12-06T10:30:45.123456"
}</pre>
                </div>
            </div>

            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Check API health and model status</p>
            </div>

            <div class="endpoint">
                <h3><span class="method">GET</span> /model-info</h3>
                <p>Get information about the loaded model</p>
            </div>

            <h3>📊 Detection Metrics</h3>
            <ul>
                <li><strong>EAR (Eye Aspect Ratio):</strong> < 0.25 indicates closed eyes</li>
                <li><strong>Mouth Ratio:</strong> > 0.6 indicates yawning</li>
                <li><strong>CNN Prediction:</strong> > 0.7 indicates drowsiness</li>
            </ul>

            <h3>🔧 Technologies</h3>
            <ul>
                <li>Flask API</li>
                <li>TensorFlow/Keras CNN</li>
                <li>MediaPipe Face Mesh</li>
                <li>OpenCV Image Processing</li>
            </ul>
        </div>
    </body>
    </html>
    '''


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'mediapipe_ready': detector.face_mesh is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if detector.model is None:
        return jsonify({
            'model_loaded': False,
            'message': 'No model loaded'
        }), 404

    try:
        model_info = {
            'model_loaded': True,
            'input_shape': detector.model.input_shape,
            'output_shape': detector.model.output_shape,
            'total_params': detector.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in detector.model.trainable_weights]),
            'layers': len(detector.model.layers),
            'optimizer': detector.model.optimizer.__class__.__name__,
            'loss': detector.model.loss,
            'metrics': detector.model.metrics_names,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(model_info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'model_loaded': True,
            'error': str(e),
            'message': 'Error retrieving model information'
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze image for drowsiness detection"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'message': 'Please provide base64 encoded image in JSON format'
            }), 400

        # Decode base64 image
        image_data = data['image']

        # Handle data URL format
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            return jsonify({
                'error': 'Invalid image data',
                'message': f'Failed to decode image: {str(e)}'
            }), 400

        # Analyze the image
        result = detector.analyze_image(opencv_image)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple images in batch"""
    try:
        data = request.get_json()

        if not data or 'images' not in data:
            return jsonify({
                'error': 'No images data provided',
                'message': 'Please provide array of base64 encoded images'
            }), 400

        images = data['images']
        if not isinstance(images, list):
            return jsonify({
                'error': 'Invalid data format',
                'message': 'Images should be provided as an array'
            }), 400

        results = []

        for i, image_data in enumerate(images):
            try:
                # Handle data URL format
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]

                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))

                # Convert PIL to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Analyze the image
                result = detector.analyze_image(opencv_image)
                result['image_index'] = i
                results.append(result)

            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': f'Failed to process image {i}: {str(e)}',
                    'face_detected': False
                })

        return jsonify({
            'total_images': len(images),
            'processed_images': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in batch-analyze endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/statistics', methods=['POST'])
def get_statistics():
    """Get statistics from analysis results"""
    try:
        data = request.get_json()

        if not data or 'results' not in data:
            return jsonify({
                'error': 'No results data provided',
                'message': 'Please provide analysis results array'
            }), 400

        results = data['results']
        if not isinstance(results, list):
            return jsonify({
                'error': 'Invalid data format',
                'message': 'Results should be provided as an array'
            }), 400

        # Calculate statistics
        total_detections = len(results)
        face_detected_count = sum(1 for r in results if r.get('face_detected', False))
        drowsy_count = sum(1 for r in results if r.get('is_drowsy', False))
        yawning_count = sum(1 for r in results if r.get('is_yawning', False))

        # Calculate averages for detected faces
        detected_results = [r for r in results if r.get('face_detected', False)]

        avg_ear = 0.0
        avg_mouth_ratio = 0.0
        avg_cnn_prediction = 0.0

        if detected_results:
            avg_ear = sum(r.get('avg_ear', 0) for r in detected_results) / len(detected_results)
            avg_mouth_ratio = sum(r.get('mouth_ratio', 0) for r in detected_results) / len(detected_results)
            avg_cnn_prediction = sum(r.get('cnn_prediction', 0) for r in detected_results) / len(detected_results)

        statistics = {
            'total_detections': total_detections,
            'face_detected_count': face_detected_count,
            'face_detection_rate': face_detected_count / total_detections if total_detections > 0 else 0,
            'drowsy_count': drowsy_count,
            'drowsy_rate': drowsy_count / face_detected_count if face_detected_count > 0 else 0,
            'yawning_count': yawning_count,
            'yawning_rate': yawning_count / face_detected_count if face_detected_count > 0 else 0,
            'average_ear': round(avg_ear, 4),
            'average_mouth_ratio': round(avg_mouth_ratio, 4),
            'average_cnn_prediction': round(avg_cnn_prediction, 4),
            'alert_level': 'HIGH' if drowsy_count / face_detected_count > 0.3 else 'MEDIUM' if drowsy_count / face_detected_count > 0.1 else 'LOW',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(statistics)

    except Exception as e:
        logger.error(f"Error in statistics endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/config', methods=['GET', 'POST'])
def configuration():
    """Get or update detection configuration"""
    if request.method == 'GET':
        return jsonify({
            'ear_threshold': 0.25,
            'yawn_threshold': 0.6,
            'cnn_threshold': 0.7,
            'drowsy_frames_threshold': 15,
            'model_loaded': detector.model is not None,
            'mediapipe_confidence': {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            }
        })

    elif request.method == 'POST':
        # Note: In a full implementation, you would update the detector configuration
        # For now, we'll just return the current configuration
        return jsonify({
            'message': 'Configuration update not implemented in this version',
            'current_config': {
                'ear_threshold': 0.25,
                'yawn_threshold': 0.6,
                'cnn_threshold': 0.7,
                'drowsy_frames_threshold': 15
            }
        })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /model-info',
            'POST /analyze',
            'POST /batch-analyze',
            'POST /statistics',
            'GET /config',
            'POST /config'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('drowsiness_model.h5'):
        logger.warning("Model file 'drowsiness_model.h5' not found. API will work with MediaPipe only.")

    # Start the Flask application
    logger.info("Starting Driver Drowsiness Detection API...")
    logger.info("Available endpoints:")
    logger.info("  GET  /           - API documentation")
    logger.info("  GET  /health     - Health check")
    logger.info("  GET  /model-info - Model information")
    logger.info("  POST /analyze    - Analyze single image")
    logger.info("  POST /batch-analyze - Analyze multiple images")
    logger.info("  POST /statistics - Get analysis statistics")
    logger.info("  GET  /config     - Get configuration")
    logger.info("  POST /config     - Update configuration")

    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )