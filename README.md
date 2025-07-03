# Drowsiness & Yawn Detection System

A modular Python system for detecting drowsiness and yawning using deep learning (Keras/TensorFlow, OpenCV, MediaPipe). Supports both classic CNN and ResNet50V2 transfer learning pipelines, with GUI and real-time camera detection.

## Features
- Classifies images into four classes: `Closed`, `Open`, `no_yawn`, `yawn`
- Two model options: classic CNN and ResNet50V2 (transfer learning)
- Modular codebase for maintainability and extensibility
- GUI and real-time camera detection modes
- Robust data loading, preprocessing, and augmentation
- Output plots and models organized in dedicated folders

## Folder Structure
```
├── analyzer.py            # Image analysis class
├── app.py                 # Real-time detection class and main application
├── gui.py                 # GUI for image-based detection
├── main.py                # Program entry point
├── models.py              # Model training (CNN & ResNet50V2)
├── utils.py               # Dataset loading & helpers
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── dataset/               # Dataset (Closed, Open, no_yawn, yawn)
├── models/                # Saved model files (.h5)
│   ├── drowsiness_cnn.h5
│   └── drowsiness_resnet50v2.h5
├── plots/                 # Output plots (.png)
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_optimized.png
│   ├── training_history_cnn.png
│   └── training_metrics_optimized.png
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare dataset:**
   - Place your images in the `dataset/` folder, organized into subfolders: `Closed`, `Open`, `no_yawn`, `yawn`.

## Usage
### 1. Training a Model
Run the main menu and select the model type to train:
```bash
python main.py
```
- Choose between classic CNN and ResNet50V2 pipelines.
- Trained models are saved in `models/` as `.h5` files.
- Training history and confusion matrix plots are saved in `plots/`.

### 2. Detection Modes
- **GUI Detection:**
  - Use the GUI to analyze static images for drowsiness/yawn.
- **Real-Time Detection:**
  - Use your webcam for live drowsiness/yawn detection.
- The system auto-detects which model is loaded and applies the correct preprocessing.

### 3. Output Files
- **Models:**
  - `models/drowsiness_cnn.h5` (classic CNN)
  - `models/drowsiness_resnet50v2.h5` (ResNet50V2)
- **Plots:**
  - `plots/training_history_cnn.png`, `plots/training_metrics_optimized.png` (training curves)
  - `plots/confusion_matrix_cnn.png`, `plots/confusion_matrix_optimized.png` (confusion matrices)

## Code Organization
- All code is modularized:
  - `main.py`: Entry point, main menu, model selection
  - `models.py`: Model training, saving, and plotting
  - `analyzer.py`: Detection classes (image & real-time)
  - `gui.py`: GUI logic
  - `utils.py`: Data loading and helper functions (optional)
- All file paths for models and plots are relative to `models/` and `plots/` folders.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies (TensorFlow, OpenCV, MediaPipe, scikit-learn, matplotlib, etc.)

## Notes
- For best results, ensure your dataset is balanced across all four classes.
- You can further tune model parameters in `models.py`.
- All outputs (models, plots) are automatically organized in their respective folders.

---

**Project maintained by [Your Name].**

