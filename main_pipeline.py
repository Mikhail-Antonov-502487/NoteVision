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
    """
    Полный конвейер распознавания нот
    """
    
    def __init__(self, debug=False):
        """
        Инициализация пайплайна
        
        Args:
            debug: Включить отладочный режим
        """
        self.debug = debug
        
        
        # Инициализация компонентов
        self.staff_remover = StaffLineRemover(debug=debug)
        self.note_extractor = NoteExtractor()
        # В __init__:
        self.note_classifier = NoteClassifier(clef='treble', debug=True)  # Включите debug=True для отладки
        self.notation_converter = NotationConverter()
        
        print("✅ Music Recognition Pipeline initialized")
    
    def process(self, image_path, save_results=True, show_results=True):
        """
        Полный цикл обработки изображения
        
        Args:
            image_path: Путь к изображению
            save_results: Сохранять ли результаты
            show_results: Показывать ли результаты
            
        Returns:
            dict: Результаты распознавания
        """
        print(f"\n{'='*60}")
        print(f"🎵 ОБРАБОТКА: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # 1. Загрузка изображения
        print("[1/5] Загрузка изображения...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        original = image.copy()
        print(f"   Размер: {image.shape[1]}x{image.shape[0]} пикселей")
        
        # 2. Удаление линий стана
        print("[2/5] Удаление линий стана...")
        staff_mask, staff_lines = self.staff_remover.extract_staff_lines(image)
        clean_image = self.staff_remover.remove_staff_lines(image, staff_mask)
        
        lines_count = len(staff_lines) if staff_lines is not None else 0
        print(f"   Найдено линий стана: {lines_count}")
        
        # 3. Извлечение нот
        print("[3/5] Поиск нот...")
        notes = self.note_extractor.extract_notes(clean_image)
        print(f"   Найдено нот: {len(notes)}")
        
        # Статистика
        stats = self.note_extractor.get_note_statistics(notes)
        if stats:
            print(f"   Заполненных: {stats.get('filled', 0)}")
            print(f"   Пустых: {stats.get('empty', 0)}")
            print(f"   Со штилем: {stats.get('with_stem', 0)}")
        
        # 4. Классификация нот
        print("[4/5] Классификация нот...")
        classified_notes = self.note_classifier.classify_all(notes, staff_lines)
        print(f"   Классифицировано: {len(classified_notes)} нот")
        
        # 5. Конвертация и сохранение
        print("[5/5] Сохранение результатов...")
        
        # Буквенная нотация
        letter_notation = self.notation_converter.to_letter_notation(classified_notes)
        print(f"\n🎵 Буквенная нотация: {letter_notation}")
        
        # Русская нотация
        russian_notation = self.notation_converter.to_russian_notation(classified_notes)
        print(f"🎵 Русская нотация: {russian_notation}")
        
        # Детальный вывод
        print("\n📋 Детали:")
        for note in classified_notes:
            print(f"  {note['id']}. {note['full_name']} ({note['russian']}) - "
                  f"{note['type_ru']} [поз.{note['position']}]")
        
        # Сохранение в файлы
        saved_files = {}
        if save_results:
            output_dir = Path('output_images')
            output_dir.mkdir(exist_ok=True)
            
            saved_files = self.notation_converter.save_all_formats(
                classified_notes, output_dir, image_path
            )
            
            # Сохраняем визуализацию
            visualization = self._create_visualization(
                original, clean_image, notes, classified_notes
            )
            
            viz_path = output_dir / f"{Path(image_path).stem}_visualization.jpg"
            cv2.imwrite(str(viz_path), visualization)
            saved_files['visualization'] = viz_path
            
            # Сохраняем чистое изображение
            clean_path = output_dir / f"{Path(image_path).stem}_clean.jpg"
            cv2.imwrite(str(clean_path), clean_image)
            saved_files['clean_image'] = clean_path
            
            print(f"\n📁 Результаты сохранены в: {output_dir}")
            for key, path in saved_files.items():
                print(f"   - {key}: {path.name}")
        
        # Показ результатов
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
        """
        Создание визуализации результатов
        """
        h, w = original.shape[:2]
        
        # Создаем большое полотно
        margin = 20
        total_width = w * 2 + margin
        total_height = h + 250
        
        # Белый фон
        result = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Левая панель - оригинал с разметкой
        left = original.copy()
        
        # Заголовок
        cv2.putText(left, "Original with Detected Notes", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 2)
        
        # Рисуем ноты
        for note_data in classified:
            # Находим соответствующий объект Note
            note_obj = None
            for n in notes:
                if n.id == note_data['id']:
                    note_obj = n
                    break
            
            if note_obj:
                x, y, w_note, h_note = note_obj.bbox
                
                # Цвет зависит от типа ноты
                if note_data['is_filled']:
                    color = (0, 0, 255)  # Красный - заполненные
                else:
                    color = (0, 255, 0)  # Зеленый - пустые
                
                # Контур ноты
                cv2.drawContours(left, [note_obj.contour], -1, color, 2)
                
                # Подпись
                label = f"{note_data['full_name']}"
                cv2.putText(left, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Центр
                cv2.circle(left, note_obj.center, 3, (0, 255, 255), -1)
        
        result[0:h, 0:w] = left
        
        # Правая панель - чистое изображение
        right = clean.copy()
        cv2.putText(right, "Clean Image (Staff Lines Removed)", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 2)
        result[0:h, w+margin:w*2+margin] = right
        
        # Нижняя панель - текстовая информация
        y_offset = h + 30
        
        # Буквенная нотация
        letter = self.notation_converter.to_letter_notation(classified)
        cv2.putText(result, "Letter Notation:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        cv2.putText(result, letter, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Русская нотация
        y_offset += 30
        russian = self.notation_converter.to_russian_notation(classified)
        cv2.putText(result, "Russian Notation:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        cv2.putText(result, russian, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Статистика
        y_offset += 30
        total = len(classified)
        filled = sum(1 for n in classified if n['is_filled'])
        cv2.putText(result, f"Total: {total} notes (Filled: {filled}, Empty: {total-filled})",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result
    
    def _show_result(self, visualization, title):
        """
        Отображение результата
        """
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title(f'Results: {title}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()



def _visualize_positions(self, image, notes, staff_lines):
    """Визуализация позиций нот на стане"""
    img = image.copy()
    
    # Рисуем линии стана
    if staff_lines is not None:
        for line in staff_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Рисуем ноты с подписями
    for note in notes:
        x, y = note.center
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        
        # Подписываем позицию
        cv2.putText(img, f"pos:{note.position}", 
                   (x - 30, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Номер ноты
        cv2.putText(img, f"#{note.id}", 
                   (x - 10, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Сохраняем отладку
    debug_dir = Path('debug_images')
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / 'positions_debug.jpg'), img)
    
    # Показываем
    cv2.imshow('Positions Debug', cv2.resize(img, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Для тестирования
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    pipeline = MusicRecognitionPipeline(debug=True)
    
    # Поиск изображений
    input_dir = Path(__file__).parent.parent / "input_images"
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if images:
        pipeline.process(images[0], save_results=True, show_results=True)
    else:
        print(f"No images found in {input_dir}")
        print("Please add images to the input_images folder")