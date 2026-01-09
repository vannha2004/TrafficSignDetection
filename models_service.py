"""
Model management and prediction services
"""
import logging
import cv2
import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ML imports
import torch
from ultralytics import YOLO
from keras.models import load_model


from config import (
    YOLO_MODEL_PATH, KERAS_MODEL_PATH, YOLO_DATA_YAML, 
    TRAFFIC_SIGNS_JSON, DEVICE
)
from utils import safe_json_load, resize_image_if_needed

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all ML models and their predictions"""
    
    def __init__(self):
        self.models = {}
        self.class_names = {}
        self.traffic_signs_dict = {}
        self._load_models()
        self._load_metadata()
    
    def _load_models(self):
        """Load all available models"""
        try:
            # Load YOLO model
            logger.info("Loading YOLO model...")
            self.models['yolo'] = YOLO(str(YOLO_MODEL_PATH))
            self.models['yolo'].to(DEVICE)
            logger.info("YOLO model loaded successfully")
            
            # Load Keras model
            logger.info("Loading Keras model...")
            self.models['keras'] = load_model(str(KERAS_MODEL_PATH))
            logger.info("Keras model loaded successfully")
            
            # Load Faster R-CNN model
            logger.info("Loading Faster R-CNN model...")
            try:
                from models.faster_rcnn.inference_example import load_faster_rcnn_model
                self.models['faster_rcnn'] = load_faster_rcnn_model()
                logger.info("Faster R-CNN model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Faster R-CNN model: {e}")
                self.models['faster_rcnn'] = None
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_metadata(self):
        """Load class names and traffic signs metadata"""
        try:
            # Load class names from YOLO data.yaml
            with open(YOLO_DATA_YAML, 'r') as f:
                data_yaml = yaml.safe_load(f)
                self.class_names = data_yaml['names']
            
            # Load traffic signs metadata
            traffic_signs_data = safe_json_load(str(TRAFFIC_SIGNS_JSON), [])
            self.traffic_signs_dict = {
                sign['code']: sign for sign in traffic_signs_data
            }
            
            logger.info(f"Loaded {len(self.class_names)} class names")
            logger.info(f"Loaded {len(self.traffic_signs_dict)} traffic sign definitions")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def get_sign_details(self, class_name: str) -> Dict[str, str]:
        """Get traffic sign details from class name"""
        return self.traffic_signs_dict.get(class_name, {
            'code': class_name,
            'name': f"Unknown sign ({class_name})",
            'description': 'No description available'
        })
    
    def predict_with_yolo(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Predict using YOLO model"""
        try:
            if 'yolo' not in self.models:
                raise ValueError("YOLO model not loaded")
            
            results = self.models['yolo'](img, verbose=False)
            detections = []
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = (self.class_names[class_id] 
                            if class_id < len(self.class_names) 
                            else f'Class_{class_id}')
                
                detections.append({
                    'class_name': class_name,
                    'confidence': float(box.conf),
                    'bbox': list(map(int, box.xyxy[0])),
                    'details': self.get_sign_details(class_name),
                    'model_used': 'YOLO v8'
                })
            
            logger.info(f"YOLO detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO prediction error: {e}")
            return []
    
    def predict_with_keras(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Predict using Keras model"""
        try:
            if 'keras' not in self.models:
                raise ValueError("Keras model not loaded")
            
            # Preprocess image for Keras
            image_resized = cv2.resize(img, (64, 64))
            image_normalized = image_resized.astype('float32') / 255.0
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            # Predict
            predictions = self.models['keras'].predict(image_batch, verbose=0)
            
            # Get the class with highest probability
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id])
            class_name = (self.class_names[class_id] 
                        if class_id < len(self.class_names) 
                        else f'Class_{class_id}')
            
            # Return full image as bbox for Keras (image classification)
            h, w = img.shape[:2]
            
            detections = [{
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [0, 0, w, h],
                'details': self.get_sign_details(class_name),
                'model_used': 'Keras CNN'
            }]
            
            logger.info(f"Keras predicted: {class_name} with confidence {confidence:.3f}")
            return detections
            
        except Exception as e:
            logger.error(f"Keras prediction error: {e}")
            return []
    
    def predict_with_faster_rcnn(self, filepath: str) -> List[Dict[str, Any]]:
        """Predict using Faster R-CNN model"""
        try:
            if self.models.get('faster_rcnn') is None:
                logger.warning("Faster R-CNN model not available")
                return []
            
            from models.faster_rcnn.inference_example import predict_with_faster_rcnn
            
            faster_rcnn_results = predict_with_faster_rcnn(
                self.models['faster_rcnn'], filepath
            )
            detections = []

            if faster_rcnn_results and len(faster_rcnn_results) > 0:
                for result in faster_rcnn_results:
                    raw_class = result.get('class', 'Unknown')
                    confidence = result.get('confidence', 0.0)
                    bbox = result.get('bbox', [0, 0, 100, 100])

                    if ':' in raw_class:
                        class_name = raw_class.split(':', 1)[1].strip()
                    else:
                        class_name = raw_class

                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'details': self.get_sign_details(class_name),
                        'model_used': 'Faster R-CNN'
                    })

            logger.info(f"Faster R-CNN detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Faster R-CNN prediction error: {e}")
            return []
    
    def predict(self, model_name: str, img: np.ndarray, filepath: str = None) -> List[Dict[str, Any]]:
        """
        Unified prediction interface
        
        Args:
            model_name: Name of the model to use
            img: Input image as numpy array
            filepath: File path (required for some models)
            
        Returns:
            List of detection results
        """
        if model_name == 'yolov12':
            return self.predict_with_yolo(img)
        elif model_name == 'keras':
            return self.predict_with_keras(img)
        elif model_name == 'fasterrcnn':
            if filepath is None:
                logger.error("Filepath required for Faster R-CNN prediction")
                return []
            return self.predict_with_faster_rcnn(filepath)
        else:
            logger.error(f"Unknown model: {model_name}")
            return []


# Global model manager instance
model_manager = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance (singleton pattern)"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
