# Drowsiness & Yawn Detection System

A modular Python system for detecting drowsiness and yawning using deep learning (Keras/TensorFlow, OpenCV, MediaPipe). Supports both classic CNN and ResNet50V2 transfer learning pipelines, with GUI and real-time camera detection.

---

## Features
- Classifies images into four classes: `Closed`, `Open`, `no_yawn`, `yawn`
- Two model options: classic CNN and ResNet50V2 (transfer learning)
- Modular codebase for maintainability and extensibility
- GUI and real-time camera detection modes
- Robust data loading, preprocessing, and augmentation
- Output plots and models organized in dedicated folders

## Folder Structure
```
├── analyzer.py            # Detection logic for images and real-time video
├── app.py                 # Real-time detection app (webcam integration)
├── gui.py                 # GUI for image-based detection (Tkinter)
├── main.py                # Program entry point, main menu, model selection
├── models.py              # Model architectures, training, saving, plotting
├── utils.py               # Dataset loading, preprocessing, helper functions
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

## Requirements
- Python 3.8+
- Webcam (for real-time mode)
- Recommended: GPU with CUDA support (for faster model training)
- All dependencies listed in `requirements.txt` (TensorFlow, OpenCV, MediaPipe, scikit-learn, matplotlib, numpy, pillow, etc.)

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare dataset:**
   - Place your images in the `dataset/` folder, organized into subfolders: `Closed`, `Open`, `no_yawn`, `yawn`.
   - Each subfolder should contain only images of the given class (e.g., `dataset/Closed/1.jpg`).
   - Recommended format: .jpg or .png, minimum size: 48x48 px.
   - Recommended number of images per class: at least 200 (the more, the better).

   Example:
   ```
   dataset/
     Closed/
       1.jpg
       2.jpg
       ...
     Open/
       1.jpg
       ...
     no_yawn/
       ...
     yawn/
       ...
   ```

## Usage
### 1. Training a Model
Run the main menu and select the model type to train:
```bash
python main.py
```
- In the menu, choose the model: classic CNN or ResNet50V2.
- Trained models are saved in `models/` as `.h5` files.
- Training history and confusion matrix plots are saved in `plots/`.
- You can modify model parameters in `models.py` (e.g., number of epochs, batch size, architecture).

### 2. Detection Modes
- **GUI Detection:**
  - Run:
    ```bash
    python gui.py
    ```
  - Use the GUI to analyze static images for drowsiness/yawn.
- **Real-Time Detection:**
  - Run:
    ```bash
    python app.py
    ```
  - Use your webcam for live drowsiness/yawn detection.
- The system auto-detects which model is loaded and applies the correct preprocessing.
- You can use your own model by copying your `.h5` file to the `models/` folder and giving it the appropriate name.

### 3. Output Files
- **Models:**
  - `models/drowsiness_cnn.h5` (classic CNN)
  - `models/drowsiness_resnet50v2.h5` (ResNet50V2)
- **Plots:**
  - `plots/training_history_cnn.png`, `plots/training_metrics_optimized.png` (training curves)
  - `plots/confusion_matrix_cnn.png`, `plots/confusion_matrix_optimized.png` (confusion matrices)

### 4. Example Output
- Example console output:
  ```
  [INFO] Training accuracy: 0.95
  [INFO] Model saved to models/drowsiness_cnn.h5
  [INFO] Confusion matrix saved to plots/confusion_matrix_cnn.png
  ```

## Code Organization
- `main.py`: Entry point, main menu, model selection
- `models.py`: Model training, saving, and plotting (architecture and hyperparameter modification possible)
- `analyzer.py`: Detection classes (image & real-time, prediction handling, preprocessing)
- `gui.py`: GUI logic (Tkinter, file selection, prediction display)
- `app.py`: Real-time detection (camera handling, live prediction)
- `utils.py`: Data loading and helper functions (dataset loading, augmentation, train/test split)

## Troubleshooting
- **No camera detected:** Make sure your camera is connected and not used by another application.
- **Import error:** Check if all dependencies are installed (`pip install -r requirements.txt`).
- **Model file missing:** Make sure you have trained a model or copied a `.h5` file to the `models/` folder.
- **Unbalanced dataset:** It is recommended to have a similar number of images in each class.

## Notes
- For best results, ensure your dataset is balanced across all four classes.
- You can further tune model parameters in `models.py`.
- All outputs (models, plots) are automatically organized in their respective folders.

## License
Educational, open-source project. You may use and modify the code for your own purposes.

---

**Project maintained by [Jakub Dziewior, Szymon Stach].**

