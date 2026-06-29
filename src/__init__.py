"""
Конфигурация проекта Music Note Reader
"""
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input_images"
    OUTPUT_DIR = BASE_DIR / "output_images"
    SRC_DIR = BASE_DIR / "src"
    
    for directory in [INPUT_DIR, OUTPUT_DIR]:
        directory.mkdir(exist_ok=True)
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    IMAGE_MAX_SIZE = (2000, 2000)
    
    STAFF_LINE_MIN_LENGTH = 100
    STAFF_LINE_MAX_GAP = 10
    HORIZONTAL_KERNEL_WIDTH = 40
    
    MIN_NOTE_AREA = 50
    MAX_NOTE_AREA = 1000
    NOTE_CIRCULARITY_MIN = 0.5
    
    VISUALIZATION_DPI = 150
    COLORS = {
        'staff_lines': (0, 255, 0),
        'note_filled': (0, 0, 255),
        'note_empty': (0, 255, 0),
        'note_center': (255, 0, 0),
        'text': (0, 0, 0),
        'beam': (255, 255, 0),
    }
    
    SHOW_PLOTS = False
    SAVE_RESULTS = True
    
    @classmethod
    def get_image_files(cls):
        if not cls.INPUT_DIR.exists():
            return []
        
        images = []
        for ext in cls.SUPPORTED_FORMATS:
            images.extend(cls.INPUT_DIR.glob(f"*{ext}"))
            images.extend(cls.INPUT_DIR.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    @classmethod
    def print_config(cls):
        print("="*50)
        print("КОНФИГУРАЦИЯ Music Note Reader")
        print("="*50)
        print(f"📁 Папка с изображениями: {cls.INPUT_DIR}")
        print(f"📁 Папка результатов: {cls.OUTPUT_DIR}")
        print(f"📏 Мин. длина линии стана: {cls.STAFF_LINE_MIN_LENGTH}")
        print(f"🎵 Мин. площадь ноты: {cls.MIN_NOTE_AREA}")
        print(f"🎵 Макс. площадь ноты: {cls.MAX_NOTE_AREA}")
        
        images = cls.get_image_files()
        print(f"\n📷 Найдено изображений: {len(images)}")
        for img in images:
            print(f"   - {img.name}")
        print("="*50)
