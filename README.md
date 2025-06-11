# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using computer vision and deep learning techniques.

## Authors
- Jakub Dziewior
- Szymon Stach

**Supervisor:** mgr inż. Krzysztof Hanzel  
**Institution:** Politechnika Śląska (Silesian University of Technology)

## Project Description

The DDD (Driver Drowsiness Detection) system is an advanced tool for real-time driver drowsiness detection. The project combines traditional image analysis methods with modern deep learning techniques, utilizing:

- **Convolutional Neural Network (CNN)** - for facial image classification
- **MediaPipe** - for facial landmark detection
- **Eye Aspect Ratio (EAR)** - for eye blink monitoring
- **Mouth Aspect Ratio** - for yawn detection

## Features

### 🔍 Drowsiness Detection
- Real-time analysis from camera feed
- Eye closure detection (EAR)
- Yawn detection
- CNN classification based on facial images

### 📊 Model Training
- Driver Drowsiness Dataset processing
- Data augmentation
- CNN model training with callbacks
- Results evaluation and visualization

### 📈 Monitoring
- Facial landmark visualization
- Real-time metrics display
- Training results logging

## System Requirements

### Environment
- Python 3.8+
- Webcam (for real-time detection)

### Dependencies
All required libraries are listed in `requirements.txt`:
pip install -r requirements.txt

### Main Libraries:
- **TensorFlow 2.13.0** - deep learning framework
- **OpenCV 4.8.0** - image processing
- **MediaPipe 0.10.5** - facial landmark detection
- **scikit-learn 1.3.0** - metrics and preprocessing
- **matplotlib/seaborn** - visualization

## Installation

1. **Clone the repository:**
git clone [repository-url]
cd driver-drowsiness-detection

2. **Create virtual environment:**
python -m venv venv
source venv/bin/activate # Linux/Mac

or

venv\Scripts\activate # Windows

3. **Install dependencies:**
pip install -r requirements.txt

4. **Download dataset (optional):**
   - Download Driver Drowsiness Dataset from Kaggle
   - Extract to `dataset/` folder
   - Structure: `dataset/Non-Drowsy/` and `dataset/Drowsy/`

## Usage

### Run the main program
python DDD.py

### Menu Options:
1. **Train new CNN model**
   - Requires dataset in `dataset/` folder
   - Automatic train/validation/test split
   - Saves model as `drowsiness_model.h5`

2. **Real-time detection test**
   - Uses webcam
   - Displays status: ALERT/DROWSY/YAWNING
   - Shows EAR metrics and CNN probability

3. **Evaluate existing model**
   - Loads saved model
   - Tests on test data
   - Generates confusion matrix and metrics

## Model Architecture

### CNN Architecture
Input (227x227x3)
├── Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
├── Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
├── Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout(0.25)
├── Conv2D(256) + BatchNorm + GlobalAvgPool + Dropout(0.5)
├── Dense(512) + BatchNorm + Dropout(0.5)
├── Dense(256) + Dropout(0.3)
└── Dense(1, sigmoid)

### Detection Metrics
- **EAR Threshold:** 0.25 (eye closure threshold)
- **Yawn Threshold:** 20% (yawning threshold)
- **Consecutive Frames:** 15 (frames needed to confirm drowsiness)

## Project Structure
driver-drowsiness-detection/
├── DDD.py # Main program file
├── requirements.txt # Dependencies
├── README.md # Documentation
├── dataset/ # Training data folder
│ ├── Non-Drowsy/ # Alert person images
│ └── Drowsy/ # Drowsy person images
├── models/ # Saved models
├── results/ # Results and plots
└── docs/ # Additional documentation

## Results

### Model Metrics
The CNN model achieves the following results on test data:
- **Accuracy:** ~95%
- **Precision:** ~94%
- **Recall:** ~96%
- **F1-Score:** ~95%

### Visualizations
The program automatically generates:
- `confusion_matrix.png` - confusion matrix
- `training_history.png` - training history
- Accuracy, loss, precision, recall plots

## Detection Algorithms

### 1. Eye Aspect Ratio (EAR)
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

Where p1-p6 are eye landmark points.

### 2. Mouth Aspect Ratio (MAR)
MAR = |vertical_distance| / |horizontal_distance|

For yawn detection.

### 3. CNN Classification
Facial image classification based on visual features.

## Configuration

### Adjustable parameters in code:
Detection thresholds
ear_threshold = 0.25 # EAR threshold
yawn_threshold = 20 # Yawn threshold (%)
drowsy_frames_threshold = 15 # Frame count

Model parameters
img_size = (227, 227) # Image size
batch_size = 32 # Batch size
epochs = 50 # Number of epochs

## Troubleshooting

### Common Issues:

1. **No camera:**
Error: Could not open camera

- Check camera connection
- Change camera index in `cv2.VideoCapture(0)`

2. **Missing dataset:**
Dataset path not found

- Download dataset from Kaggle
- Set correct path in `DATASET_PATH`

3. **CUDA errors:**
- Install TensorFlow-GPU if you have NVIDIA card
- Or use CPU version
