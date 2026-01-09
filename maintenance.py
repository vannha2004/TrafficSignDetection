"""
Maintenance and cleanup utilities for the Traffic Sign Detection application
"""
import logging
import os
import time
from pathlib import Path
from typing import Optional

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_FILE
from utils import cleanup_old_files

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """Handles application maintenance tasks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cleanup_upload_files(self, max_files: int = 100):
        """Clean up old uploaded files"""
        try:
            self.logger.info(f"Cleaning up upload folder: {UPLOAD_FOLDER}")
            cleanup_old_files(str(UPLOAD_FOLDER), max_files)
            self.logger.info("Upload folder cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning upload folder: {e}")
    
    def cleanup_output_files(self, max_files: int = 50):
        """Clean up old output files"""
        try:
            self.logger.info(f"Cleaning up output folder: {OUTPUT_FOLDER}")
            cleanup_old_files(str(OUTPUT_FOLDER), max_files)
            self.logger.info("Output folder cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning output folder: {e}")
    
    def cleanup_log_files(self, max_size_mb: int = 10):
        """Clean up log files if they get too large"""
        try:
            if not LOG_FILE.exists():
                return
            
            # Check log file size
            size_mb = LOG_FILE.stat().st_size / (1024 * 1024)
            
            if size_mb > max_size_mb:
                # Backup current log
                backup_path = LOG_FILE.with_suffix(f'.backup.{int(time.time())}.log')
                LOG_FILE.rename(backup_path)
                
                self.logger.info(f"Log file rotated. Backup saved as: {backup_path}")
                
                # Keep only last 3 backup files
                backup_files = sorted(LOG_FILE.parent.glob('*.backup.*.log'))
                for old_backup in backup_files[:-3]:
                    old_backup.unlink()
                    self.logger.info(f"Removed old backup: {old_backup}")
                    
        except Exception as e:
            self.logger.error(f"Error managing log files: {e}")
    
    def get_disk_usage(self) -> dict:
        """Get disk usage information for important directories"""
        dirs_to_check = {
            'upload': UPLOAD_FOLDER,
            'output': OUTPUT_FOLDER,
            'logs': LOG_FILE.parent
        }
        
        usage_info = {}
        
        for name, path in dirs_to_check.items():
            try:
                if Path(path).exists():
                    # Calculate directory size
                    total_size = sum(
                        f.stat().st_size for f in Path(path).rglob('*') if f.is_file()
                    )
                    
                    # Count files
                    file_count = len([f for f in Path(path).rglob('*') if f.is_file()])
                    
                    usage_info[name] = {
                        'size_mb': round(total_size / (1024 * 1024), 2),
                        'file_count': file_count,
                        'path': str(path)
                    }
                else:
                    usage_info[name] = {
                        'size_mb': 0,
                        'file_count': 0,
                        'path': str(path)
                    }
                    
            except Exception as e:
                self.logger.error(f"Error calculating usage for {name}: {e}")
                usage_info[name] = {
                    'size_mb': -1,
                    'file_count': -1,
                    'path': str(path),
                    'error': str(e)
                }
        
        return usage_info
    
    def run_full_maintenance(self):
        """Run all maintenance tasks"""
        self.logger.info("Starting full maintenance cycle")
        
        self.cleanup_upload_files()
        self.cleanup_output_files() 
        self.cleanup_log_files()
        
        usage = self.get_disk_usage()
        self.logger.info(f"Disk usage after cleanup: {usage}")
        
        self.logger.info("Full maintenance cycle completed")


def run_maintenance_cli():
    """CLI function to run maintenance tasks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run maintenance tasks')
    parser.add_argument('--uploads', action='store_true', help='Clean upload files')
    parser.add_argument('--outputs', action='store_true', help='Clean output files')
    parser.add_argument('--logs', action='store_true', help='Clean log files')
    parser.add_argument('--all', action='store_true', help='Run all maintenance tasks')
    parser.add_argument('--status', action='store_true', help='Show disk usage status')
    
    args = parser.parse_args()
    
    # Setup basic logging for CLI
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    manager = MaintenanceManager()
    
    if args.status or not any([args.uploads, args.outputs, args.logs, args.all]):
        print("=== Disk Usage Status ===")
        usage = manager.get_disk_usage()
        for name, info in usage.items():
            print(f"{name.capitalize()}: {info['size_mb']} MB, {info['file_count']} files")
        print()
    
    if args.all:
        manager.run_full_maintenance()
    else:
        if args.uploads:
            manager.cleanup_upload_files()
        if args.outputs:
            manager.cleanup_output_files()
        if args.logs:
            manager.cleanup_log_files()


if __name__ == '__main__':
    run_maintenance_cli()
