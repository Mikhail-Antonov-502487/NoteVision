"""
Конфигурация проекта Music Note Reader
"""
from pathlib import Path
import os

class Config:
    """Основные настройки проекта"""
    
    # Пути
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input_images"
    OUTPUT_DIR = BASE_DIR / "output_images"
    SRC_DIR = BASE_DIR / "src"
    
    # Создание папок при запуске
    for directory in [INPUT_DIR, OUTPUT_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Поддерживаемые форматы изображений
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Параметры обработки изображений
    IMAGE_MAX_SIZE = (2000, 2000)  # Максимальный размер для обработки
    
    # Параметры для удаления линий стана
    STAFF_LINE_MIN_LENGTH = 100
    STAFF_LINE_MAX_GAP = 10
    HORIZONTAL_KERNEL_WIDTH = 40
    
    # Параметры для поиска нот
    MIN_NOTE_AREA = 50
    MAX_NOTE_AREA = 1000
    NOTE_CIRCULARITY_MIN = 0.5
    
    # Параметры визуализации
    VISUALIZATION_DPI = 150
    COLORS = {
        'staff_lines': (0, 255, 0),      # Зеленый
        'note_filled': (0, 0, 255),       # Красный
        'note_empty': (0, 255, 0),        # Зеленый
        'note_center': (255, 0, 0),       # Синий
        'text': (0, 0, 0),                # Черный
        'beam': (255, 255, 0),            # Желтый
    }
    
    # Настройки вывода
    SHOW_PLOTS = False  # Показывать ли графики при обработке
    SAVE_RESULTS = True  # Сохранять ли результаты
    
    @classmethod
    def get_image_files(cls):
        """Получить список всех изображений в INPUT_DIR"""
        if not cls.INPUT_DIR.exists():
            return []
        
        images = []
        for ext in cls.SUPPORTED_FORMATS:
            images.extend(cls.INPUT_DIR.glob(f"*{ext}"))
            images.extend(cls.INPUT_DIR.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    @classmethod
    def print_config(cls):
        """Вывод текущей конфигурации"""
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