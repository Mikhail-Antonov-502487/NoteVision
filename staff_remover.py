"""
Модуль удаления линий нотоносца
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import Config

class StaffLineRemover:
    """
    Класс для обнаружения и удаления линий нотоносца
    """
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def extract_staff_lines(self, image):
        """
        Извлечение линий нотоносца из изображения
        
        Args:
            image: BGR изображение или бинарная маска
            
        Returns:
            staff_mask: Бинарная маска линий стана
            lines: Найденные линии
        """
        # Если пришло цветное изображение, конвертируем в бинарное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Адаптивная бинаризация для лучшего выделения линий
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )
        else:
            binary = image.copy()
        
        # Создаем ядро для выделения горизонтальных линий
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (Config.HORIZONTAL_KERNEL_WIDTH, 1)
        )
        
        # Выделяем горизонтальные линии
        horizontal_lines = cv2.morphologyEx(
            binary, 
            cv2.MORPH_OPEN, 
            horizontal_kernel
        )
        
        # Удаляем вертикальные компоненты
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Оставляем только горизонтальные линии
        staff_mask = cv2.subtract(horizontal_lines, vertical_lines)
        
        # Очистка маски
        kernel = np.ones((3,3), np.uint8)
        staff_mask = cv2.morphologyEx(staff_mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск линий с помощью HoughLinesP
        lines = cv2.HoughLinesP(
            staff_mask,
            1,                    # Разрешение по расстоянию
            np.pi/180,           # Разрешение по углу
            threshold=50,         # Порог
            minLineLength=Config.STAFF_LINE_MIN_LENGTH,
            maxLineGap=Config.STAFF_LINE_MAX_GAP
        )
        
        if self.debug:
            self._show_extraction_steps(binary, horizontal_lines, staff_mask)
        
        return staff_mask, lines
    
    def remove_staff_lines(self, image, staff_mask):
        """
        Удаление линий стана с изображения
        
        Args:
            image: Исходное цветное изображение
            staff_mask: Маска линий стана
            
        Returns:
            clean_image: Изображение без линий стана
        """
        result = image.copy()
        
        # Расширяем маску для лучшего покрытия линий
        kernel = np.ones((3,3), np.uint8)
        staff_mask_dilated = cv2.dilate(staff_mask, kernel, iterations=1)
        
        # Инвертируем маску
        staff_mask_inv = cv2.bitwise_not(staff_mask_dilated)
        
        # Применяем маску ко всем каналам
        for channel in range(3):
            result[:, :, channel] = cv2.bitwise_and(
                result[:, :, channel],
                staff_mask_inv
            )
        
        # Восстановление изображения на месте удаленных линий
        result = cv2.inpaint(
            result,
            staff_mask_dilated,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )
        
        if self.debug:
            self._show_removal_result(image, staff_mask, result)
        
        return result
    
    def _show_extraction_steps(self, binary, horizontal, staff_mask):
        """Визуализация шагов извлечения линий"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(binary, cmap='gray')
        axes[0].set_title('Бинаризованное изображение')
        axes[0].axis('off')
        
        axes[1].imshow(horizontal, cmap='gray')
        axes[1].set_title('Горизонтальные линии')
        axes[1].axis('off')
        
        axes[2].imshow(staff_mask, cmap='gray')
        axes[2].set_title('Маска линий стана')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _show_removal_result(self, original, mask, result):
        """Визуализация результата удаления линий"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Маска удаляемых линий')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title('После удаления линий')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()