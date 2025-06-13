from enum import Enum

class FileType(str, Enum):
    CSV = 'text/csv'
    TXT = 'text/plain'
    PDF = 'application/pdf'
