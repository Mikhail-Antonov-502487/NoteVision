"""
Главный конвейер обработки изображений
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .staff_remover import StaffLineRemover
from .note_extractor import NoteExtractor
from .note_classifier import NoteClassifier
from .notation_converter import NotationConverter


class MusicRecognitionPipeline:
    def __init__(self, debug=False):
        self.debug = debug
        self.staff_remover = StaffLineRemover(debug=debug)
        self.note_extractor = NoteExtractor()
        self.note_classifier = NoteClassifier(clef='treble', debug=True)
        self.notation_converter = NotationConverter()
        print("✅ Music Recognition Pipeline initialized")
    
    def process(self, image_path, save_results=True, show_results=True):
        print(f"\n{'='*60}")
        print(f"🎵 ОБРАБОТКА: {Path(image_path).name}")
        print(f"{'='*60}")
        
        print("[1/5] Загрузка изображения...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        original = image.copy()
        print(f"   Размер: {image.shape[1]}x{image.shape[0]} пикселей")
        
        print("[2/5] Удаление линий стана...")
        staff_mask, staff_lines = self.staff_remover.extract_staff_lines(image)
        clean_image = self.staff_remover.remove_staff_lines(image, staff_mask)
        
        lines_count = len(staff_lines) if staff_lines is not None else 0
        print(f"   Найдено линий стана: {lines_count}")
        
        print("[3/5] Поиск нот...")
        notes = self.note_extractor.extract_notes(clean_image)
        print(f"   Найдено нот: {len(notes)}")
        
        stats = self.note_extractor.get_note_statistics(notes)
        if stats:
            print(f"   Заполненных: {stats.get('filled', 0)}")
            print(f"   Пустых: {stats.get('empty', 0)}")
            print(f"   Со штилем: {stats.get('with_stem', 0)}")
        
        print("[4/5] Классификация нот...")
        classified_notes = self.note_classifier.classify_all(notes, staff_lines)
        print(f"   Классифицировано: {len(classified_notes)} нот")
        
        print("[5/5] Сохранение результатов...")
        
        letter_notation = self.notation_converter.to_letter_notation(classified_notes)
        print(f"\n🎵 Буквенная нотация: {letter_notation}")
        
        russian_notation = self.notation_converter.to_russian_notation(classified_notes)
        print(f"🎵 Русская нотация: {russian_notation}")
        
        print("\n📋 Детали:")
        for note in classified_notes:
            print(f"  {note['id']}. {note['full_name']} ({note['russian']}) - "
                  f"{note['type_ru']} [поз.{note['position']}]")
        
        saved_files = {}
        if save_results:
            output_dir = Path('output_images')
            output_dir.mkdir(exist_ok=True)
            
            saved_files = self.notation_converter.save_all_formats(
                classified_notes, output_dir, image_path
            )
            
            visualization = self._create_visualization(
                original, clean_image, notes, classified_notes
            )
            
            viz_path = output_dir / f"{Path(image_path).stem}_visualization.jpg"
            cv2.imwrite(str(viz_path), visualization)
            saved_files['visualization'] = viz_path
            
            clean_path = output_dir / f"{Path(image_path).stem}_clean.jpg"
            cv2.imwrite(str(clean_path), clean_image)
            saved_files['clean_image'] = clean_path
            
            print(f"\n📁 Результаты сохранены в: {output_dir}")
            for key, path in saved_files.items():
                print(f"   - {key}: {path.name}")
        
        if show_results and 'visualization' in saved_files:
            self._show_result(visualization, Path(image_path).name)
        
        print(f"{'='*60}\n")
        
        return {
            'image_path': image_path,
            'original': original,
            'clean_image': clean_image,
            'staff_mask': staff_mask,
            'staff_lines': staff_lines,
            'notes': notes,
            'classified_notes': classified_notes,
            'letter_notation': letter_notation,
            'russian_notation': russian_notation,
            'saved_files': saved_files,
            'visualization': saved_files.get('visualization')
        }
    
    def _create_visualization(self, original, clean, notes, classified):
        h, w = original.shape[:2]
        
        margin = 20
        total_width = w * 2 + margin
        total_height = h + 250
        
        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        left = original.copy()
        cv2.putText(left, "Original with Detected Notes", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6
