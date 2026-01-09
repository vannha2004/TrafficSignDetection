"""
Detection history management service
"""
import threading
import time
import logging
from typing import List, Dict, Any
from collections import deque

from config import HISTORY_JSON, MAX_HISTORY_SIZE, HISTORY_SAVE_INTERVAL
from utils import safe_json_load, safe_json_save, create_detection_history_entry

logger = logging.getLogger(__name__)


class HistoryManager:
    """Manages detection history with thread-safe operations"""
    
    def __init__(self):
        self.history = deque(maxlen=MAX_HISTORY_SIZE)
        self.lock = threading.Lock()
        self._load_history()
        self._start_background_saver()
    
    def _load_history(self):
        """Load existing history from file"""
        try:
            existing_history = safe_json_load(str(HISTORY_JSON), [])
            with self.lock:
                self.history.extend(existing_history[-MAX_HISTORY_SIZE:])
            logger.info(f"Loaded {len(self.history)} history entries")
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    def _start_background_saver(self):
        """Start background thread to periodically save history"""
        def save_periodically():
            while True:
                try:
                    self.save_to_file()
                    time.sleep(HISTORY_SAVE_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in background history saver: {e}")
                    time.sleep(HISTORY_SAVE_INTERVAL)
        
        thread = threading.Thread(target=save_periodically, daemon=True)
        thread.start()
        logger.info("Started background history saver")
    
    def add_detection(
        self, 
        class_name: str, 
        model_used: str, 
        confidence: float = 0.0,
        additional_data: Dict = None
    ):
        try:
            entry = create_detection_history_entry(
                class_name, model_used, confidence, additional_data
            )
            
            with self.lock:
                self.history.append(entry)
            
            logger.debug(f"Added detection to history: {class_name}")
            
        except Exception as e:
            logger.error(f"Error adding detection to history: {e}")
    
    def add_detections(self, detections: List[Dict[str, Any]]):
        """
        Add multiple detections to history
        
        Args:
            detections: List of detection results
        """
        for detection in detections:
            self.add_detection(
                class_name=detection.get('class_name', 'Unknown'),
                model_used=detection.get('model_used', 'Unknown'),
                confidence=detection.get('confidence', 0.0)
            )
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent history entries
        
        Args:
            limit: Number of entries to return
            
        Returns:
            List of recent history entries
        """
        with self.lock:
            return list(self.history)[-limit:]
    
    def get_all_history(self) -> List[Dict[str, Any]]:
        """Get all history entries"""
        with self.lock:
            return list(self.history)
    
    def save_to_file(self) -> bool:
        """
        Save current history to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                history_list = list(self.history)
            
            success = safe_json_save(history_list, str(HISTORY_JSON))
            if success:
                logger.debug(f"Saved {len(history_list)} history entries to file")
            return success
            
        except Exception as e:
            logger.error(f"Error saving history to file: {e}")
            return False
    
    def clear_history(self):
        """Clear all history entries"""
        with self.lock:
            self.history.clear()
        logger.info("Cleared all history entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detection history
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            history_list = list(self.history)
        
        if not history_list:
            return {
                'total_detections': 0,
                'unique_signs': 0,
                'models_used': [],
                'most_detected': None,
                'average_confidence': 0.0
            }
        
        # Count detections by type
        sign_counts = {}
        model_counts = {}
        confidences = []
        
        for entry in history_list:
            sign_type = entry.get('type', 'Unknown')
            model = entry.get('model', 'Unknown')
            confidence = entry.get('confidence', 0.0)
            
            sign_counts[sign_type] = sign_counts.get(sign_type, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1
            
            if confidence > 0:
                confidences.append(confidence)
        
        # Find most detected sign
        most_detected = max(sign_counts.items(), key=lambda x: x[1]) if sign_counts else None
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'total_detections': len(history_list),
            'unique_signs': len(sign_counts),
            'models_used': list(model_counts.keys()),
            'most_detected': most_detected,
            'average_confidence': avg_confidence,
            'sign_distribution': sign_counts,
            'model_distribution': model_counts
        }


# Global history manager instance
history_manager = None


def get_history_manager() -> HistoryManager:
    """Get global history manager instance (singleton pattern)"""
    global history_manager
    if history_manager is None:
        history_manager = HistoryManager()
    return history_manager
