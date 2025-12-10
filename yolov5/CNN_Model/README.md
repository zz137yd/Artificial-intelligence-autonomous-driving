# 🏎️ Steering Angle Estimation with Bottom-View CNN (PyTorch)

This project implements a lightweight CNN model that predicts steering angles from only the bottom 100 pixels of an image, suitable for self-driving applications.

---

## 📁 Project Structure

```
├── config.py               # Configuration file
├── model.py                # CNN regression model
├── utils.py                # Dataset class and CSV generator
├── train.py                # Training script (basic)
├── train_show.py           # Training with preview visualization
├── inference.py            # Inference from image/video/folder
├── check.py                # Model parameter shape checker
├── dataset/
│   ├── images/             # Input images (*.jpg)
│   ├── labels.txt          # Steering angles (1 per image)
│   └── labels.csv          # Auto-generated label CSV
└── weights/
    └── best_model.pth      # Trained model example
```

---

## 🧠 Model Overview

- **Input**: RGB image (320x180)
- **Region Used**: Bottom 100 pixels only (80–180)
- **Output**: Single float (steering angle regression)
- **Architecture**:
  - 5× Conv2D + ReLU
  - Flatten → 4× Linear
- **Loss Function**: MSELoss

---

## 🧰 Environment Setup

Python 3.8+ and dependencies:

```bash
pip install torch torchvision opencv-python pandas matplotlib Pillow tqdm
```

---

## 📌 Usage

### ① Generate CSV (only once)

```bash
python train.py
```

### ② Train the Model

```bash
python train.py
```

Or for preview-enabled training:

```bash
python train_show.py
```

Checkpoints and logs will be saved under `./train/expN/`.

### ③ Inference

```bash
python inference.py --input <path> --model <path_to_model>
```

Examples:
- Single image: `--input image.jpg`
- Video: `--input video.mp4`
- Folder: `--input ./images/`
- Disable saving: `--no-save`

Output saved to `./inference/expN/`.

### ④ Check Model Parameters

```bash
python check.py
```

---

## ⚙️ Configuration (config.py)

```python
DATASET_DIR = './dataset/images'
LABELS_TXT = './dataset/labels.txt'
LABELS_CSV = './dataset/labels.csv'

RESIZE_HEIGHT = 180
RESIZE_WIDTH = 320
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 100
```

---

## 🖼️ Visualization Highlights

- Top 80 pixels: Gaussian blur
- Bottom 100 pixels: Red rectangle
- Predicted angle: Drawn on image/video

---

## 📎 Notes

- Steering angle units depend on label input (deg or rad)
- Auto GPU (CUDA) support
- Optimized for real-time inference
- Supported input formats: `.jpg`, `.mp4`, `.avi`

---

© 2025 SteeringNet Project