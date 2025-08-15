"""Local file storage service."""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from app.config import get_settings


class StorageService:
    """Local file storage service."""
    
    def __init__(self):
        """Initialize storage service."""
        self.settings = get_settings()
        self.upload_dir = Path(self.settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
    
    def put_object(self, file_content: bytes, filename: str, content_type: Optional[str] = None) -> str:
        """Save file content to local storage."""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_ext = Path(filename).suffix
            unique_filename = f"{file_id}{file_ext}"
            file_path = self.upload_dir / unique_filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            return unique_filename
        except Exception as e:
            raise Exception(f"Failed to save file: {e}")
    
    def get_object(self, filename: str) -> bytes:
        """Read file content from local storage."""
        try:
            file_path = self.upload_dir / filename
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read file: {e}")
    
    def get_file_path(self, filename: str) -> str:
        """Get full file path."""
        return str(self.upload_dir / filename)
    
    def delete_object(self, filename: str) -> bool:
        """Delete file from local storage."""
        try:
            file_path = self.upload_dir / filename
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to delete file: {e}")
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists."""
        file_path = self.upload_dir / filename
        return file_path.exists()


# Global storage instance
storage = StorageService()
