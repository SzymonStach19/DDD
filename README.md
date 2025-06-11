# Drowsiness Detection System

## Description
A robust system for drowsiness detection based on face and eye images. Uses classic CNN or ResNet50V2 (Keras/TensorFlow) and MediaPipe Face Mesh to detect closed eyes and yawning in photos and in real-time from a camera. Supports both GUI and real-time camera detection, and allows the user to choose between model types at startup.

## Features
- Single photo analysis (GUI or CLI)
- Real-time drowsiness detection from camera
- Train your own model on your own dataset
- Supports four classes: closed eyes, open eyes, yawning, no yawning
- Choice of model: classic CNN (grayscale, 150x150) or ResNet50V2 (color, 224x224)
- Automatic preprocessing and model compatibility
- Training history and confusion matrix plots

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- Keras
- scikit-learn
- Pillow
- MediaPipe
- Tkinter (for GUI)
- matplotlib

It is recommended to install all dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Directory Structure
```
├── DDD.py                      # Main program file (all logic, menu, models, GUI, detection)
├── drowsiness_cnn.h5           # Trained classic CNN model (if exists)
├── drowsiness_resnet50v2.h5    # Trained ResNet50V2 model (if exists)
├── training_history_cnn.png    # Training history plot for CNN
├── training_metrics_optimized.png # Training history plot for ResNet50V2
├── confusion_matrix_cnn.png    # Confusion matrix for CNN
├── confusion_matrix_optimized.png # Confusion matrix for ResNet50V2
├── requirements.txt            # List of dependencies
├── dataset/                    # Data folder (subfolders: Closed, Open, yawn, no_yawn)
```

## Usage

### 1. Model Training
If you don't have a model, just run the program. The model will be trained automatically using data from the `dataset/` folder. You will be prompted to choose the model type (classic CNN or ResNet50V2).

```bash
python DDD.py
```

### 2. Photo Analysis (GUI)
Select option 1 in the menu, then choose a photo for analysis. You can select detection mode (eyes, yawning, or both) in the GUI.

### 3. Real-time Detection (Camera)
Select option 2 in the menu, choose detection mode, and observe live results from your camera. You can switch detection mode during runtime using keys 1 (eyes), 2 (yawning), 3 (both).

### 4. Photo Analysis (CLI)
Select option 3 in the menu, provide the path to the photo and detection mode.

## Detection Modes
- Eyes only – only eyes
- Yawning only – only yawning
- Eyes and yawning – both at once

## Notes
- The program will prompt you to train a new model or use an existing one if found.
- All code and logic is in `DDD.py`.
- The GUI and real-time detection work with both model types.

## Authors
Jakub Dziewior
Szymon Stach

