"""
Configuration settings for Traffic Sign Detection application
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'upload'
OUTPUT_FOLDER = BASE_DIR / 'static' / 'output'

# Model paths
MODELS_DIR = BASE_DIR / 'models'
YOLO_MODEL_PATH = MODELS_DIR / 'yolo' / 'best.pt'
KERAS_MODEL_PATH = MODELS_DIR / 'keras' / 'best_model.h5'
FASTER_RCNN_MODEL_PATH = MODELS_DIR / 'faster_rcnn' / 'best_model.pth'
FASTER_RCNN_CONFIG_PATH = MODELS_DIR / 'faster_rcnn' / 'training_config.py'

# Data files
YOLO_DATA_YAML = MODELS_DIR / 'yolo' / 'data.yaml'
TRAFFIC_SIGNS_JSON = MODELS_DIR / 'traffic_signs.json'
HISTORY_JSON = BASE_DIR / 'detection_history.json'

# Application settings
FLASK_DEBUG = True
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000

# Detection settings
MAX_HISTORY_SIZE = 1000
HISTORY_SAVE_INTERVAL = 5  # seconds
DEFAULT_MODEL = 'yolov12'

# Image processing
MAX_IMAGE_SIZE = (1024, 1024)
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Available models configuration
AVAILABLE_MODELS = {
    'yolov12': {
        'name': 'YOLO v12',
        'description': 'Fast object detection with good accuracy',
        'model_path': YOLO_MODEL_PATH
    },
    'keras': {
        'name': 'Keras CNN',
        'description': 'Image classification model',
        'model_path': KERAS_MODEL_PATH
    },
    'fasterrcnn': {
        'name': 'Faster R-CNN',
        'description': 'High accuracy object detection',
        'model_path': FASTER_RCNN_MODEL_PATH,
        'config_path': FASTER_RCNN_CONFIG_PATH
    }
}

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / 'app.log'

# Device settings
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
