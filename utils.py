"""
Utility functions for the Traffic Sign Detection application
"""
import logging
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from config import LOG_FILE, LOG_FORMAT, LOG_LEVEL


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_image_file(filepath: str) -> bool:
    """
    Validate if the file is a valid image
    
    Args:
        filepath: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        img = cv2.imread(filepath)
        return img is not None
    except Exception:
        return False


def safe_json_load(filepath: str, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or invalid
        
    Returns:
        Loaded JSON data or default value
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default or []
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading JSON file {filepath}: {e}")
        return default or []


def safe_json_save(data: Any, filepath: str) -> bool:
    """
    Safely save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, TypeError) as e:
        logging.error(f"Error saving JSON file {filepath}: {e}")
        return False


def create_detection_history_entry(
    class_name: str, 
    model_used: str, 
    confidence: float = 0.0,
    additional_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a standardized detection history entry
    
    Args:
        class_name: Detected class name
        model_used: Model that made the detection
        confidence: Detection confidence score
        additional_data: Additional metadata
        
    Returns:
        Dictionary with detection history entry
    """
    entry = {
        'type': class_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_used,
        'confidence': confidence
    }
    
    if additional_data:
        entry.update(additional_data)
        
    return entry


def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()


def is_supported_image_format(filename: str) -> bool:
    """Check if file format is supported"""
    from config import SUPPORTED_IMAGE_FORMATS
    return get_file_extension(filename) in SUPPORTED_IMAGE_FORMATS


def resize_image_if_needed(img: np.ndarray, max_size: tuple = (1024, 1024)) -> np.ndarray:
    """
    Resize image if it's larger than max_size while maintaining aspect ratio
    
    Args:
        img: Input image
        max_size: Maximum size (width, height)
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    max_w, max_h = max_size
    
    if w <= max_w and h <= max_h:
        return img
    
    # Calculate scaling factor
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string"""
    return f"{confidence * 100:.1f}%"


def cleanup_old_files(directory: str, max_files: int = 100):
    """
    Clean up old files in directory, keeping only the most recent ones
    
    Args:
        directory: Directory to clean
        max_files: Maximum number of files to keep
    """
    try:
        files = list(Path(directory).glob('*'))
        if len(files) <= max_files:
            return
            
        # Sort by modification time, oldest first
        files.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest files
        for file in files[:-max_files]:
            try:
                file.unlink()
                logging.info(f"Cleaned up old file: {file}")
            except OSError as e:
                logging.error(f"Error removing file {file}: {e}")
                
    except Exception as e:
        logging.error(f"Error during cleanup of {directory}: {e}")
