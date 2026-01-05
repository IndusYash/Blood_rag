"""
File Handler module for Medical Report AI
Handles file uploads, validation, and storage
"""

import os
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from typing import Optional, Dict
import shutil
from datetime import datetime

from config import Config


class FileHandler:
    """Handle file operations for medical reports"""
    
    def __init__(self, upload_folder: Path = None):
        """
        Initialize file handler
        
        Args:
            upload_folder: Path to upload folder
        """
        self.upload_folder = upload_folder or Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self.max_file_size = Config.MAX_CONTENT_LENGTH
        
        # Create upload folder if it doesn't exist
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def is_allowed_file(self, filename: str) -> bool:
        """
        Check if file extension is allowed
        
        Args:
            filename: Name of the file
            
        Returns:
            True if allowed, False otherwise
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def validate_file(self, file: FileStorage) -> Dict:
        """
        Validate uploaded file
        
        Args:
            file: Uploaded file object
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        
        # Check if file exists
        if not file or file.filename == '':
            errors.append("No file provided")
            return {'valid': False, 'errors': errors}
        
        # Check file extension
        if not self.is_allowed_file(file.filename):
            errors.append(f"File type not allowed. Allowed: {', '.join(self.allowed_extensions)}")
        
        # Check file size (if possible)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > self.max_file_size:
            errors.append(f"File too large. Max size: {self.max_file_size / (1024*1024):.1f}MB")
        
        if file_size == 0:
            errors.append("File is empty")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'file_size': file_size,
            'filename': file.filename
        }
    
    def save_uploaded_file(self, file: FileStorage, custom_filename: str = None) -> str:
        """
        Save uploaded file to disk
        
        Args:
            file: Uploaded file object
            custom_filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        # Validate file
        validation = self.validate_file(file)
        if not validation['valid']:
            raise ValueError(f"File validation failed: {', '.join(validation['errors'])}")
        
        # Generate unique filename
        if custom_filename:
            filename = secure_filename(custom_filename)
        else:
            original_filename = secure_filename(file.filename)
            file_ext = Path(original_filename).suffix
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{unique_id}{file_ext}"
        
        # Create user-specific subfolder (optional)
        save_folder = self.upload_folder
        os.makedirs(save_folder, exist_ok=True)
        
        # Save file
        file_path = save_folder / filename
        file.save(str(file_path))
        
        return str(file_path)
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            file_path = Path(file_path)
            if file_path.exists() and file_path.is_file():
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting file: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get information about a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None
            
            stat_info = file_path.stat()
            
            return {
                'filename': file_path.name,
                'path': str(file_path),
                'size': stat_info.st_size,
                'size_mb': stat_info.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat_info.st_ctime),
                'modified': datetime.fromtimestamp(stat_info.st_mtime),
                'extension': file_path.suffix
            }
        except Exception as e:
            print(f"Error getting file info: {e}")
            return None
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Delete files older than specified days
        
        Args:
            days: Number of days
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            current_time = datetime.now()
            
            for file_path in self.upload_folder.glob('*'):
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_age.days > days:
                        os.remove(file_path)
                        deleted_count += 1
            
            return deleted_count
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            return 0
    
    def get_upload_stats(self) -> Dict:
        """
        Get statistics about uploaded files
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            files = list(self.upload_folder.glob('*'))
            files = [f for f in files if f.is_file()]
            
            total_size = sum(f.stat().st_size for f in files)
            
            # Count by extension
            extensions = {}
            for f in files:
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
            
            return {
                'total_files': len(files),
                'total_size': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'by_extension': extensions,
                'upload_folder': str(self.upload_folder)
            }
        except Exception as e:
            print(f"Error getting upload stats: {e}")
            return {}


def main():
    """Test file handler"""
    print("File Handler Test")
    print("=" * 50)
    
    handler = FileHandler()
    
    print(f"\nUpload folder: {handler.upload_folder}")
    print(f"Allowed extensions: {handler.allowed_extensions}")
    print(f"Max file size: {handler.max_file_size / (1024*1024):.1f}MB")
    
    # Get upload stats
    stats = handler.get_upload_stats()
    print(f"\nUpload Statistics:")
    print(f"Total files: {stats.get('total_files', 0)}")
    print(f"Total size: {stats.get('total_size_mb', 0):.2f}MB")


if __name__ == "__main__":
    main()
