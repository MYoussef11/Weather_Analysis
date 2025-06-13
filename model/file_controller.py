import os
from typing import List


class FileController:
    def __init__(self):
        self.allowed_types = eval(os.getenv("FILE_ALLOWED_TYPES", "['text/csv']"))
        self.max_size_mb = int(os.getenv("FILE_MAX_SIZE", "10"))
        self.max_size_bytes = self.max_size_mb * 1024 * 1024

    def is_allowed_filetype(self, mimetype: str) -> bool:
        """Check if the file mimetype is allowed."""
        return mimetype in self.allowed_types

    def is_allowed_filesize(self, size_bytes: int) -> bool:
        """Check if the file size is within the allowed limit."""
        return size_bytes <= self.max_size_bytes

    def validate_upload(self, mimetype: str, size_bytes: int) -> list:
        """Validate file upload and return a list of warnings if any."""
        warnings = []
        if not self.is_allowed_filetype(mimetype):
            warnings.append(f"File type '{mimetype}' is not allowed. Allowed types: {self.allowed_types}")
        if not self.is_allowed_filesize(size_bytes):
            warnings.append(f"File size {size_bytes / (1024*1024):.2f} MB exceeds the limit of {self.max_size_mb} MB.")
        return warnings
