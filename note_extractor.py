"""
Модуль извлечения и анализа нот
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Note:
    """Структура данных для ноты"""
    id: int
    contour: np.ndarray
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    area: float
    perimeter: float
    circularity: float
    is_filled: bool
    has_stem: bool
    stem_direction: str
    position: int = 0
    ellipse: Optional[Tuple] = None

class NoteExtractor:
    """Класс для извлечения и анализа нотных головок"""
    
    def __init__(self, staff_lines=None):
        self.min_note_area = 30
        self.max_note_area = 1500
        self.min_ellipse_ratio = 0.20
        self.max_ellipse_ratio = 0.95
        self.min_extent = 0.20
        self.max_extent = 0.95
        self.min_solidity = 0.35
        self.note_counter = 0
        self.staff_lines = staff_lines
        self.staff_bounds = None
        self.image_width = 0
        
    def extract_notes(self, image, staff_lines=None):
        """Извлечение всех нот из изображения"""
        if staff_lines is not None:
            self.staff_lines = staff_lines
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        self.image_width = gray.shape[1]
        
        # Вычисляем границы станов
        self._calculate_staff_bounds()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        notes = []
        self.note_counter = 0
        
        for contour in contours:
            note = self._analyze_contour(contour, cleaned)
            if note is not None:
                self.note_counter += 1
                note.id = self.note_counter
                notes.append(note)
        
        notes.sort(key=lambda n: n.center[0])
        return notes
    
    def _calculate_staff_bounds(self):
        """Вычисление границ станов"""
        if self.staff_lines is None or len(self.staff_lines) == 0:
            self.staff_bounds = None
            return
        
        all_y = []
        for line in self.staff_lines:
            if len(line) > 0:
                x1, y1, x2, y2 = line[0]
                avg_y = (y1 + y2) / 2
                all_y.append(avg_y)
        
        if len(all_y) < 5:
            self.staff_bounds = None
            return
        
        all_y = sorted(all_y)
        
        # Группируем линии в станы
        groups = []
        current_group = [all_y[0]]
        
        for y in all_y[1:]:
            if y - current_group[-1] > 30:
                if len(current_group) >= 5:
                    groups.append(current_group)
                current_group = [y]
            else:
                current_group.append(y)
        
        if len(current_group) >= 5:
            groups.append(current_group)
        
        # Для каждой группы вычисляем границы
        self.staff_bounds = []
        for group in groups:
            if len(group) >= 5:
                step = (group[-1] - group[0]) / 4
                margin = step * 3.0
                top = group[0] - margin
                bottom = group[-1] + margin
                self.staff_bounds.append((top, bottom))
    
    def _is_valid_position(self, x, y):
        """
        Проверка, находится ли объект в допустимой позиции
        """
        # ❗ ФИЛЬТР 1: Отсеиваем текст в левом углу
        # Ноты обычно начинаются не с самого левого края
        # Определяем динамический порог в зависимости от ширины изображения
        left_margin = self.image_width * 0.08  # 8% от ширины
        if x < left_margin:
            return False
        
        # ❗ ФИЛЬТР 2: Проверка, что объект находится в пределах стана
        if self.staff_bounds is not None and len(self.staff_bounds) > 0:
            is_within_staff = False
            for top, bottom in self.staff_bounds:
                if top <= y <= bottom:
                    is_within_staff = True
                    break
            
            if not is_within_staff:
                return False
        
        return True
    
    def _analyze_contour(self, contour, binary_image):
        """Анализ контура на предмет соответствия ноте"""
        area = cv2.contourArea(contour)
        
        if area < self.min_note_area or area > self.max_note_area:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < 4 or h < 4:
            return None
        
        center_x = x + w/2
        center_y = y + h/2
        
        # ❗ ГЛАВНЫЙ ФИЛЬТР: Проверка позиции
        if not self._is_valid_position(center_x, center_y):
            return None
        
        # Базовая проверка соотношения сторон
        ratio = w / float(h) if h > 0 else 0
        if ratio < 0.3 or ratio > 2.5:
            return None
        
        # Проверка на эллиптическую форму
        if len(contour) < 5:
            return None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            major_axis, minor_axis = axes
            
            if major_axis == 0 or minor_axis == 0:
                return None
            
            ellipse_ratio = min(major_axis, minor_axis) / max(major_axis, minor_axis)
            
            if ellipse_ratio < 0.20 or ellipse_ratio > 0.95:
                return None
            
            if major_axis < 4 or minor_axis < 3:
                return None
            
        except:
            return None
        
        # Проверка плотности заполнения
        extent = area / float(w * h) if w * h > 0 else 0
        if extent < 0.20 or extent > 0.95:
            return None
        
        # Проверка компактности
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None
        solidity = area / hull_area
        if solidity < 0.35:
            return None
        
        # Центр масс
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Определяем заполненность
        is_filled = self._check_if_filled(contour, binary_image)
        
        # Поиск штиля
        has_stem, stem_direction = self._detect_stem(contour, binary_image, x, y, w, h)
        
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return Note(
            id=0,
            contour=contour,
            center=(cx, cy),
            bbox=(x, y, w, h),
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            is_filled=is_filled,
            has_stem=has_stem,
            stem_direction=stem_direction,
            ellipse=ellipse
        )
    
    def _check_if_filled(self, contour, binary_image) -> bool:
        mask = np.zeros(binary_image.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        inside = cv2.bitwise_and(binary_image, binary_image, mask=mask)
        filled = cv2.countNonZero(inside)
        total = cv2.countNonZero(mask)
        
        if total == 0:
            return False
        
        ratio = filled / total
        return ratio > 0.35
    
    def _detect_stem(self, contour, binary_image, x, y, w, h) -> Tuple[bool, str]:
        right_region = self._get_stem_region(binary_image, x, y, w, h, 'right')
        has_right_stem = self._check_stem_presence(right_region)
        
        if has_right_stem:
            return True, 'up'
        
        left_region = self._get_stem_region(binary_image, x, y, w, h, 'left')
        has_left_stem = self._check_stem_presence(left_region)
        
        if has_left_stem:
            return True, 'down'
        
        return False, 'none'
    
    def _get_stem_region(self, binary, x, y, w, h, side='right'):
        height, width = binary.shape
        
        if side == 'right':
            x1 = min(width-1, x + w)
            x2 = min(width-1, x + w + w)
        else:
            x1 = max(0, x - w)
            x2 = max(0, x)
        
        y1 = max(0, y - h * 2)
        y2 = min(height-1, y + h * 3)
        
        if x2 <= x1 or y2 <= y1:
            return np.array([])
        
        return binary[y1:y2, x1:x2]
    
    def _check_stem_presence(self, region) -> bool:
        if region.size == 0:
            return False
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
        vertical_lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, vertical_kernel)
        
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            _, _, _, h = cv2.boundingRect(cnt)
            if h > 10:
                return True
        
        return False
    
    def get_note_statistics(self, notes: List[Note]) -> dict:
        if not notes:
            return {}
        
        stats = {
            'total': len(notes),
            'filled': sum(1 for n in notes if n.is_filled),
            'empty': sum(1 for n in notes if not n.is_filled),
            'with_stem': sum(1 for n in notes if n.has_stem),
            'without_stem': sum(1 for n in notes if not n.has_stem),
            'avg_area': np.mean([n.area for n in notes]),
            'avg_circularity': np.mean([n.circularity for n in notes])
        }
        
        return stats