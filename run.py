#!/usr/bin/env python3
"""
Music Note Reader - Простой запуск
"""
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from staff_remover import StaffLineRemover
from note_extractor import NoteExtractor
from note_classifier import NoteClassifier
from notation_converter import NotationConverter


def safe_imread(image_path):
    try:
        image = cv2.imread(str(image_path))
        if image is not None:
            return image
        
        with open(image_path, 'rb') as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        
        from PIL import Image
        pil_img = Image.open(image_path)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None


def process_image(image_path, show=True, debug=False):
    print(f"\n{'='*50}")
    print(f"Обработка: {Path(image_path).name}")
    print(f"{'='*50}")
    
    image = safe_imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return None
    
    original = image.copy()
    
    print("1. Удаление линий стана...")
    remover = StaffLineRemover(debug=False)
    staff_mask, staff_lines = remover.extract_staff_lines(image)
    clean_image = remover.remove_staff_lines(image, staff_mask)
    
    print("2. Поиск нот...")
    extractor = NoteExtractor(staff_lines=staff_lines)
    notes = extractor.extract_notes(clean_image, staff_lines=staff_lines)
    print(f"   Найдено нот: {len(notes)}")
    
    if len(notes) == 0:
        print("❌ Ноты не найдены!")
        return None
    
    print("3. Классификация нот...")
    classifier = NoteClassifier(clef='treble', debug=debug)
    classified = classifier.classify_all(notes, staff_lines)
    
    print("4. Создание нотации...")
    converter = NotationConverter()
    
    letter_notation = " ".join([n['letter'] for n in classified])
    russian_notation = " ".join([n['russian'] for n in classified])
    
    print(f"\n🎵 Буквенная: {letter_notation}")
    print(f"🎵 Русская:   {russian_notation}")
    
    print("\n📋 Детали:")
    for note in classified:
        print(f"  {note['id']}. {note['letter']} ({note['russian']}) - {note['type_ru']}")
    
    output_dir = Path('output_images')
    output_dir.mkdir(exist_ok=True)
    converter.save_all_formats(classified, output_dir, image_path)
    
    if show and notes:
        result = original.copy()
        note_data_map = {n['id']: n for n in classified}
        
        for note in notes:
            if note.id in note_data_map:
                note_data = note_data_map[note.id]
                color = (0, 0, 255) if note_data['is_filled'] else (255, 100, 0)
                
                if hasattr(note, 'ellipse') and note.ellipse is not None:
                    cv2.ellipse(result, note.ellipse, color, 2)
                else:
                    cv2.drawContours(result, [note.contour], -1, color, 2)
                
                cv2.circle(result, note.center, 3, (255, 0, 255), -1)
        
        for note in notes:
            if note.id in note_data_map:
                note_data = note_data_map[note.id]
                x, y, w, h = note.bbox
                label = note_data['letter']
                
                text_x = x + w // 2 - 8
                text_y = y - 8
                if text_y < 20:
                    text_y = y + h + 20
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                
                cv2.rectangle(result, 
                            (text_x - 3, text_y - th - 3),
                            (text_x + tw + 3, text_y + 3),
                            (255, 255, 255), -1)
                cv2.putText(result, label, (text_x, text_y),
                           font, 0.6, (0, 0, 255), 2)
        
        result_path = output_dir / f"{Path(image_path).stem}_result.jpg"
        cv2.imwrite(str(result_path), result)
        print(f"\n📁 Результат сохранен: {result_path}")
        
        plt.figure(figsize=(14, 10))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Результат: {Path(image_path).name}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return classified


def main():
    input_dir = Path('input_images')
    input_dir.mkdir(exist_ok=True)
    
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(input_dir.glob(ext))
    
    if not images:
        print("❌ Нет изображений в папке input_images/")
        return
    
    print("\n📷 Найдены изображения:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")
    
    print("\nВыберите:")
    print(f"  1-{len(images)} - номер изображения")
    print("  a    - обработать все")
    print("  d    - обработать с отладкой")
    print("  q    - выход")
    
    choice = input("> ").strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'a':
        for img in images:
            try:
                process_image(img, show=False)
            except Exception as e:
                print(f"Ошибка: {e}")
    elif choice == 'd':
        if images:
            process_image(images[0], debug=True)
    elif choice.isdigit() and 1 <= int(choice) <= len(images):
        process_image(images[int(choice)-1])
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
