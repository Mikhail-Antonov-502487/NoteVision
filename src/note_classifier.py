"""
Модуль классификации нот и определения их высоты
"""
import numpy as np
from typing import List

class NoteClassifier:
    TREBLE_CLEF = {
        -4: ('A', 'Ля'),
        -3: ('B', 'Си'),
        -2: ('C', 'До'),
        -1: ('D', 'Ре'),
        0:  ('E', 'Ми'),
        1:  ('F', 'Фа'),
        2:  ('G', 'Соль'),
        3:  ('A', 'Ля'),
        4:  ('B', 'Си'),
        5:  ('C', 'До'),
        6:  ('D', 'Ре'),
        7:  ('E', 'Ми'),
        8:  ('F', 'Фа'),
        9:  ('G', 'Соль'),
        10: ('A', 'Ля'),
        11: ('B', 'Си'),
        12: ('C', 'До'),
        13: ('D', 'Ре'),
        14: ('E', 'Ми'),
        15: ('F', 'Фа'),
        16: ('G', 'Соль'),
    }
    
    def __init__(self, clef='treble', debug=False):
        self.clef = clef
        self.debug = debug
        self.position_map = self.TREBLE_CLEF
    
    def determine_position(self, note, staff_lines):
        if staff_lines is None or len(staff_lines) == 0:
            return 4
        
        note_y = note.center[1]
        
        line_ys = []
        for line in staff_lines:
            if len(line) > 0:
                x1, y1, x2, y2 = line[0]
                avg_y = (y1 + y2) / 2
                line_ys.append(avg_y)
        
        if len(line_ys) < 5:
            return 4
        
        distances = [(abs(note_y - y), y) for y in line_ys]
        distances.sort()
        closest_lines = sorted([d[1] for d in distances[:5]])
        
        step = closest_lines[1] - closest_lines[0]
        if step < 3:
            step = 10
        
        closest_idx = np.argmin([abs(note_y - y) for y in closest_lines])
        closest_y = closest_lines[closest_idx]
        y_diff = note_y - closest_y
        
        if abs(y_diff) < step * 0.4:
            position = closest_idx * 2
        elif y_diff > 0:
            if closest_idx < 4:
                position = closest_idx * 2 + 1
            else:
                extra = int(abs(y_diff) / step) + 1
                position = 8 + extra * 2
        else:
            if closest_idx > 0:
                position = (closest_idx - 1) * 2 + 1
            else:
                extra = int(abs(y_diff) / step) + 1
                position = -extra * 2
        
        return position
    
    def get_note_name(self, position):
        if position in self.position_map:
            letter, russian = self.position_map[position]
        else:
            if position > 16:
                letter, russian = 'G', 'Соль'
            elif position < -4:
                letter, russian = 'A', 'Ля'
            else:
                letter, russian = 'C', 'До'
        
        return {
            "letter": letter,
            "russian": russian,
            "full_name": letter,
        }
    
    def classify_note_type(self, note, has_beam=False, beam_count=0):
        if not note.is_filled and not note.has_stem:
            return 'whole'
        elif not note.is_filled and note.has_stem:
            return 'half'
        elif note.is_filled and note.has_stem and not has_beam:
            return 'quarter'
        elif note.is_filled and note.has_stem and has_beam:
            if beam_count >= 2:
                return 'sixteenth'
            else:
                return 'eighth'
        else:
            return 'quarter'
    
    def classify_all(self, notes, staff_lines, beam_groups=None):
        NOTE_TYPES = {
            'whole':      {'ru': 'Целая',      'duration': 4.0},
            'half':       {'ru': 'Половинная', 'duration': 2.0},
            'quarter':    {'ru': 'Четвертная', 'duration': 1.0},
            'eighth':     {'ru': 'Восьмая',    'duration': 0.5},
            'sixteenth':  {'ru': 'Шестнадцатая','duration': 0.25},
        }
        
        classified = []
        
        beamed_notes = set()
        beam_counts = {}
        
        if beam_groups:
            for group in beam_groups:
                if len(group) > 1:
                    for note in group:
                        beamed_notes.add(id(note))
                        beam_counts[id(note)] = len(group) - 1
        
        for note in notes:
            position = self.determine_position(note, staff_lines)
            
            has_beam = id(note) in beamed_notes
            beam_count = beam_counts.get(id(note), 0)
            note_type = self.classify_note_type(note, has_beam, beam_count)
            note_name = self.get_note_name(position)
            
            note_info = {
                'id': int(note.id),
                'type': note_type,
                'type_ru': NOTE_TYPES[note_type]['ru'],
                'duration': float(NOTE_TYPES[note_type]['duration']),
                'letter': note_name['letter'],
                'russian': note_name['russian'],
                'full_name': note_name['full_name'],
                'position': int(position),
                'is_filled': bool(note.is_filled),
                'has_stem': bool(note.has_stem),
                'has_beam': bool(has_beam),
                'center': (int(note.center[0]), int(note.center[1])),
                'bbox': [int(x) for x in note.bbox]
            }
            
            if self.debug:
                print(f"Нота {note.id}: позиция {position} -> {note_name['letter']} ({note_name['russian']})")
            
            classified.append(note_info)
        
        return sorted(classified, key=lambda n: n['center'][0])
