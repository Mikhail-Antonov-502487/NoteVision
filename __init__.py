"""
Music Note Reader - Распознавание нот из изображений
"""

from .staff_remover import StaffLineRemover
from .note_extractor import NoteExtractor
from .note_classifier import NoteClassifier
from .notation_converter import NotationConverter

# Импортируем pipeline только если все зависимости доступны
try:
    from .main_pipeline import MusicRecognitionPipeline
    __all__ = [
        'StaffLineRemover',
        'NoteExtractor', 
        'NoteClassifier',
        'NotationConverter',
        'MusicRecognitionPipeline'
    ]
except ImportError as e:
    print(f"Warning: Could not import MusicRecognitionPipeline: {e}")
    __all__ = [
        'StaffLineRemover',
        'NoteExtractor', 
        'NoteClassifier',
        'NotationConverter'
    ]

__version__ = "1.0.0"
__author__ = "Your Name"