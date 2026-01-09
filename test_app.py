"""
Basic tests for the Traffic Sign Detection application
"""
import unittest
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Import our modules
from utils import (
    validate_image_file, is_supported_image_format, 
    safe_json_load, safe_json_save, create_detection_history_entry
)
from models_service import ModelManager
from history_service import HistoryManager


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_is_supported_image_format(self):
        """Test image format validation"""
        self.assertTrue(is_supported_image_format('test.jpg'))
        self.assertTrue(is_supported_image_format('test.PNG'))
        self.assertFalse(is_supported_image_format('test.txt'))
        self.assertFalse(is_supported_image_format('test.doc'))
    
    def test_create_detection_history_entry(self):
        """Test detection history entry creation"""
        entry = create_detection_history_entry('P-102', 'YOLO v8', 0.95)
        
        self.assertEqual(entry['type'], 'P-102')
        self.assertEqual(entry['model'], 'YOLO v8')
        self.assertEqual(entry['confidence'], 0.95)
        self.assertIn('timestamp', entry)
    
    def test_safe_json_operations(self):
        """Test JSON save and load operations"""
        test_data = {'test': 'data', 'number': 123}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            result = safe_json_save(test_data, temp_path)
            self.assertTrue(result)
            
            # Test load
            loaded_data = safe_json_load(temp_path)
            self.assertEqual(loaded_data, test_data)
            
            # Test load non-existent file
            non_existent = safe_json_load('/non/existent/path.json', default=[])
            self.assertEqual(non_existent, [])
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestHistoryManager(unittest.TestCase):
    """Test history management"""
    
    def setUp(self):
        """Set up test environment"""
        self.history_manager = HistoryManager()
        self.history_manager.clear_history()  # Start with clean slate
    
    def test_add_detection(self):
        """Test adding detection to history"""
        initial_count = len(self.history_manager.get_all_history())
        
        self.history_manager.add_detection('P-102', 'YOLO v8', 0.95)
        
        new_count = len(self.history_manager.get_all_history())
        self.assertEqual(new_count, initial_count + 1)
        
        recent = self.history_manager.get_recent_history(1)
        self.assertEqual(recent[0]['type'], 'P-102')
    
    def test_get_statistics(self):
        """Test statistics generation"""
        # Add some test detections
        self.history_manager.add_detection('P-102', 'YOLO v8', 0.95)
        self.history_manager.add_detection('P-102', 'YOLO v8', 0.87)
        self.history_manager.add_detection('DP-135', 'Keras', 0.92)
        
        stats = self.history_manager.get_statistics()
        
        self.assertEqual(stats['total_detections'], 3)
        self.assertEqual(stats['unique_signs'], 2)
        self.assertIn('YOLO v8', stats['models_used'])
        self.assertIn('Keras', stats['models_used'])


class TestImageValidation(unittest.TestCase):
    """Test image validation functions"""
    
    def test_validate_image_file(self):
        """Test image file validation"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save test image
            cv2.imwrite(temp_path, test_image)
            
            # Test validation
            self.assertTrue(validate_image_file(temp_path))
            
            # Test with non-existent file
            self.assertFalse(validate_image_file('/non/existent/image.jpg'))
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestModelManagerMock(unittest.TestCase):
    """Test model manager functionality (without loading actual models)"""
    
    def test_get_sign_details(self):
        """Test sign details retrieval"""
        # This test would need a mock or test model manager
        # For now, we'll skip actual model loading tests
        pass
    
    def test_model_prediction_interface(self):
        """Test prediction interface structure"""
        # This would test the prediction interface without loading models
        pass


def create_test_image(width=100, height=100, filename='test_image.jpg'):
    """Helper function to create test images"""
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(filename, image)
    return filename


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestHistoryManager))
    test_suite.addTest(unittest.makeSuite(TestImageValidation))
    test_suite.addTest(unittest.makeSuite(TestModelManagerMock))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
