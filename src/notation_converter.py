"""
Модуль конвертации результатов в различные форматы
"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class NotationConverter:
    def __init__(self):
        self.output_formats = ['txt', 'json', 'letter', 'russian']
    
    def _convert_to_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj
    
    def to_letter_notation(self, classified_notes):
        return " ".join([n['letter'] for n in classified_notes])
    
    def to_russian_notation(self, classified_notes):
        return " ".join([n['russian'] for n in classified_notes])
    
    def to_detailed_text(self, classified_notes, image_path=None):
        lines = []
        lines.append("=" * 60)
        lines.append("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ НОТ")
        lines.append("=" * 60)
        
        if image_path:
            lines.append(f"Файл: {Path(image_path).name}")
        
        lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Всего нот: {len(classified_notes)}")
        lines.append("-" * 60)
        lines.append(f"{'№':<4} {'Нота':<6} {'Рус.':<8} {'Тип':<15} {'Длит.':<8}")
        lines.append("-" * 60)
        
        for note in classified_notes:
            lines.append(
                f"{note['id']:<4} "
                f"{note['letter']:<6} "
                f"{note['russian']:<8} "
                f"{note['type_ru']:<15} "
                f"{note['duration']:<8}"
            )
        
        lines.append("-" * 60)
        lines.append(f"\nБуквенная нотация: {self.to_letter_notation(classified_notes)}")
        lines.append(f"Русская нотация: {self.to_russian_notation(classified_notes)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def to_json(self, classified_notes, image_path=None):
        notes_data = []
        for note in classified_notes:
            note_entry = {
                'id': self._convert_to_serializable(note['id']),
                'letter': note['letter'],
                'russian': note['russian'],
                'type': note['type'],
                'type_russian': note['type_ru'],
                'duration': self._convert_to_serializable(note['duration']),
                'position': self._convert_to_serializable(note['position']),
                'is_filled': self._convert_to_serializable(note.get('is_filled', False)),
                'has_stem': self._convert_to_serializable(note.get('has_stem', False)),
                'center_x': self._convert_to_serializable(note['center'][0]),
                'center_y': self._convert_to_serializable(note['center'][1])
            }
            notes_data.append(note_entry)
        
        data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'image': str(Path(image_path).name) if image_path else None,
                'total_notes': len(classified_notes)
            },
            'letter_notation': self.to_letter_notation(classified_notes),
            'russian_notation': self.to_russian_notation(classified_notes),
            'notes': notes_data
        }
        
        return json.dumps(data, ensure_ascii=False, indent=2, default=self._convert_to_serializable)
    
    def save_all_formats(self, classified_notes, output_dir, image_path=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        saved_files = {}
        
        txt_path = output_dir / 'results.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.to_detailed_text(classified_notes, image_path))
        saved_files['txt'] = txt_path
        
        json_path = output_dir / 'results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json(classified_notes, image_path))
        saved_files['json'] = json_path
        
        letter_path = output_dir / 'letter_notation.txt'
        with open(letter_path, 'w', encoding='utf-8') as f:
            f.write(self.to_letter_notation(classified_notes))
        saved_files['letter'] = letter_path
        
        russian_path = output_dir / 'russian_notation.txt'
        with open(russian_path, 'w', encoding='utf-8') as f:
            f.write(self.to_russian_notation(classified_notes))
        saved_files['russian'] = russian_path
        
        return saved_files
