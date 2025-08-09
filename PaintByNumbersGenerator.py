import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import os


class PaintByNumbersGenerator:
    def __init__(self, n_colors=16, blur_value=1, quantization_factor=4, max_image_size=2000, contour_thickness=2):
        """
        Инициализация генератора картин по номерам

        Args:
            n_colors (int): Количество цветов (красок) в палитре
            blur_value (int): Сила размытия для сглаживания
            quantization_factor (int): Фактор квантизации для уменьшения деталей
            max_image_size (int): Максимальный размер изображения для обработки
            contour_thickness (int): Толщина контуров (1-10, рекомендуется 1-4)
        """
        self.n_colors = n_colors
        self.blur_value = blur_value
        self.quantization_factor = quantization_factor
        self.max_image_size = max_image_size
        self.contour_thickness = contour_thickness
        self.color_palette = None
        self.labels = None
        self.original_shape = None

    def preprocess_image(self, image_path):
        """Предварительная обработка изображения"""
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удается загрузить изображение: {image_path}")

        # Конвертируем BGR в RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.original_shape = image.shape

        # Изменяем размер изображения, если оно слишком большое
        height, width = image.shape[:2]
        if max(height, width) > self.max_image_size:
            scale_factor = self.max_image_size / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Изображение изменено с {width}x{height} до {new_width}x{new_height}")

        # Применяем размытие для сглаживания мелких деталей
        if self.blur_value > 0:
            image = cv2.GaussianBlur(image, (self.blur_value*2+1, self.blur_value*2+1), 0)

        # Квантизация для уменьшения количества уникальных цветов
        image = image // self.quantization_factor * self.quantization_factor

        return image

    def cluster_colors(self, image):
        """Кластеризация цветов изображения"""
        # Преобразуем изображение в массив пикселей
        pixels = image.reshape(-1, 3)

        # Используем MiniBatchKMeans для больших изображений (быстрее)
        if len(pixels) > 100000:
            kmeans = MiniBatchKMeans(n_clusters=self.n_colors, random_state=42, batch_size=1000)
        else:
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)

        # Выполняем кластеризацию
        labels = kmeans.fit_predict(pixels)

        # Сохраняем результаты
        self.labels = labels.reshape(image.shape[:2])
        self.color_palette = kmeans.cluster_centers_.astype(int)

        return self.labels, self.color_palette

    def create_regions(self, labels):
        """Создание связанных областей одного цвета (память-эффективная версия)"""
        regions = {}
        processed_labels = np.zeros_like(labels, dtype=np.int32)
        region_id = 0

        # Минимальный размер области (адаптивный к размеру изображения)
        min_area = max(50, (labels.shape[0] * labels.shape[1]) // 10000)

        for color_id in range(self.n_colors):
            # Создаем маску для текущего цвета
            color_mask = (labels == color_id).astype(np.uint8)

            # Проверяем, есть ли пиксели этого цвета
            if not np.any(color_mask):
                continue

            # Находим связанные компоненты
            num_labels, labeled_image = cv2.connectedComponents(color_mask)

            # Обрабатываем каждую компоненту
            for component_id in range(1, num_labels):
                # Находим координаты компоненты без создания полной маски
                component_coords = np.where(labeled_image == component_id)

                # Фильтруем слишком маленькие области
                area = len(component_coords[0])
                if area < min_area:
                    continue

                # Вычисляем центроид напрямую из координат
                centroid_y = int(np.mean(component_coords[0]))
                centroid_x = int(np.mean(component_coords[1]))

                # Вычисляем bounding box для оптимизации
                min_y, max_y = np.min(component_coords[0]), np.max(component_coords[0])
                min_x, max_x = np.min(component_coords[1]), np.max(component_coords[1])

                # Сохраняем только необходимую информацию (без полной маски)
                regions[region_id] = {
                    'coordinates': component_coords,  # Храним координаты вместо маски
                    'color_id': color_id,
                    'color': self.color_palette[color_id],
                    'centroid': (centroid_x, centroid_y),
                    'area': area,
                    'bbox': (min_x, min_y, max_x, max_y)
                }

                # Обновляем processed_labels
                processed_labels[component_coords] = region_id
                region_id += 1

            # Освобождаем память
            del labeled_image, color_mask

        return regions, processed_labels

    def create_contours_optimized(self, regions, image_shape):
        """Создание контуров областей с настраиваемой толщиной"""
        contour_image = np.ones(image_shape[:2], dtype=np.uint8) * 255

        for region_id, region_data in regions.items():
            coordinates = region_data['coordinates']
            bbox = region_data['bbox']
            min_x, min_y, max_x, max_y = bbox

            # Работаем только с областью bounding box
            roi_height = max_y - min_y + 1
            roi_width = max_x - min_x + 1

            if roi_height <= 0 or roi_width <= 0:
                continue

            # Создаем маленькую маску только для текущей области
            small_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)

            # Заполняем маску (координаты относительно ROI)
            rel_y = coordinates[0] - min_y
            rel_x = coordinates[1] - min_x

            # Проверяем границы
            valid_indices = (rel_y >= 0) & (rel_y < roi_height) & (rel_x >= 0) & (rel_x < roi_width)
            small_mask[rel_y[valid_indices], rel_x[valid_indices]] = 255

            # Находим контуры в маленькой области
            contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Переносим контуры обратно в полное изображение
            for contour in contours:
                contour[:, :, 0] += min_x  # смещение по X
                contour[:, :, 1] += min_y  # смещение по Y

                # Рисуем контур с настраиваемой толщиной
                cv2.drawContours(contour_image, [contour], -1, 0, self.contour_thickness)

        return contour_image

    def create_clustered_image(self, regions, labels, image_shape):
        """Создание кластеризованного изображения без черных пятен"""
        clustered_image = np.zeros(image_shape, dtype=np.uint8)

        # Сначала заполняем базовое изображение цветами из кластеризации
        for color_id in range(self.n_colors):
            color_mask = (labels == color_id)
            clustered_image[color_mask] = self.color_palette[color_id]

        # Затем перезаписываем области больших регионов для лучшего качества
        for region_id, region_data in regions.items():
            coordinates = region_data['coordinates']
            color = region_data['color']

            # Заполняем пиксели цветом региона
            clustered_image[coordinates] = color

        return clustered_image

    def upscale_to_original_size(self, image, target_shape):
        """Увеличение изображения до оригинального размера"""
        target_height, target_width = target_shape[:2]
        current_height, current_width = image.shape[:2]

        # Проверяем, нужно ли масштабирование
        if current_height == target_height and current_width == target_width:
            return image

        # Используем INTER_NEAREST для четких контуров (без размытия)
        if len(image.shape) == 2:  # Grayscale
            upscaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        else:  # RGB
            upscaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        return upscaled

    def upscale_pil_image(self, pil_array, target_shape):
        """Увеличение PIL изображения до оригинального размера"""
        target_height, target_width = target_shape[:2]
        current_height, current_width = pil_array.shape[:2]

        if current_height == target_height and current_width == target_width:
            return pil_array

        # Конвертируем в PIL Image для качественного масштабирования текста
        pil_image = Image.fromarray(pil_array)

        # Используем LANCZOS для лучшего качества текста при увеличении
        upscaled_pil = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return np.array(upscaled_pil)

    def calculate_adaptive_font_size(self, area, image_shape):
        """Вычисление адаптивного размера шрифта для области"""
        # Базовый размер шрифта в зависимости от размера изображения
        base_font_size = max(8, min(image_shape[:2]) // 60)

        # Адаптируем к размеру области
        # Используем квадратный корень для более плавного масштабирования
        area_factor = np.sqrt(area / 1000.0)  # нормализуем относительно средней области
        adaptive_size = int(base_font_size * np.clip(area_factor, 0.5, 2.5))

        return max(8, min(adaptive_size, 40))  # ограничиваем диапазо

    def find_best_label_position(self, region_data, text_size, image_shape, occupied_areas):
        """Поиск лучшей позиции для размещения номера"""
        centroid_x, centroid_y = region_data['centroid']
        bbox = region_data['bbox']
        min_x, min_y, max_x, max_y = bbox
        text_width, text_height = text_size

        # Список потенциальных позиций (в порядке приоритета)
        positions_to_try = [
            # Центр области
            (centroid_x - text_width // 2, centroid_y - text_height // 2),
            # Центр верх
            (centroid_x - text_width // 2, min_y + 5),
            # Центр низ
            (centroid_x - text_width // 2, max_y - text_height - 5),
            # Левый центр
            (min_x + 5, centroid_y - text_height // 2),
            # Правый центр
            (max_x - text_width - 5, centroid_y - text_height // 2),
            # Углы области
            (min_x + 5, min_y + 5),
            (max_x - text_width - 5, min_y + 5),
            (min_x + 5, max_y - text_height - 5),
            (max_x - text_width - 5, max_y - text_height - 5),
        ]

        padding = 3
        for pos_x, pos_y in positions_to_try:
            # Проверяем границы изображения
            if (pos_x < 0 or pos_y < 0 or
                pos_x + text_width >= image_shape[1] or
                pos_y + text_height >= image_shape[0]):
                continue

            # Создаем прямоугольник для текста с отступами
            text_rect = (
                pos_x - padding,
                pos_y - padding,
                pos_x + text_width + padding,
                pos_y + text_height + padding
            )

            # Проверяем пересечение с уже размещенными номерами
            collision = False
            for occupied_rect in occupied_areas:
                if self.rectangles_overlap(text_rect, occupied_rect):
                    collision = True
                    break

            if not collision:
                # Проверяем, что позиция находится внутри области
                if self.point_in_region(pos_x + text_width // 2,
                                      pos_y + text_height // 2,
                                      region_data):
                    return pos_x, pos_y, text_rect

        # Если не нашли идеальную позицию, возвращаем центроид
        return (centroid_x - text_width // 2,
                centroid_y - text_height // 2,
                (centroid_x - text_width // 2 - padding,
                 centroid_y - text_height // 2 - padding,
                 centroid_x + text_width // 2 + padding,
                 centroid_y + text_height // 2 + padding))

    def rectangles_overlap(self, rect1, rect2):
        """Проверка пересечения двух прямоугольников"""
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2

        return not (x1_max < x2_min or x2_max < x1_min or
                   y1_max < y2_min or y2_max < y1_min)

    def point_in_region(self, x, y, region_data):
        """Проверка, находится ли точка внутри области"""
        coordinates = region_data['coordinates']

        # Простая проверка через bbox (можно улучшить)
        min_x, min_y, max_x, max_y = region_data['bbox']
        return min_x <= x <= max_x and min_y <= y <= max_y

    def create_contour_template(self, regions, image_shape):
        """Создание шаблона с умным размещением номеров"""
        # Создаем белый фон
        contour_template = np.ones((*image_shape[:2], 3), dtype=np.uint8) * 255

        # Добавляем контуры
        contour_lines = self.create_contours_optimized(regions, image_shape)
        contour_rgb = cv2.cvtColor(contour_lines, cv2.COLOR_GRAY2RGB)
        contour_template = np.where(contour_rgb == 0, 0, contour_template)

        # Конвертируем в PIL для работы с текстом
        contour_template_pil = Image.fromarray(contour_template)
        draw = ImageDraw.Draw(contour_template_pil)

        # Список занятых областей для предотвращения пересечений
        occupied_areas = []

        # Сортируем регионы: сначала большие, потом маленькие
        sorted_regions = sorted(regions.items(), key=lambda x: x[1]['area'], reverse=True)

        # Минимальная площадь для показа номера (адаптивная)
        total_image_area = image_shape[0] * image_shape[1]
        min_area_threshold = max(200, total_image_area // 5000)

        for region_id, region_data in sorted_regions:
            color_id = region_data['color_id']
            area = region_data['area']

            # Пропускаем слишком маленькие области
            if area < min_area_threshold:
                continue

            # Адаптивный размер шрифта
            font_size = self.calculate_adaptive_font_size(area, image_shape)

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Текст номера краски
            text = str(color_id + 1)

            # Размер текста
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Находим лучшую позицию
            pos_x, pos_y, text_rect = self.find_best_label_position(
                region_data, (text_width, text_height), image_shape, occupied_areas
            )

            # Проверяем финальные границы
            if (pos_x < 0 or pos_y < 0 or
                pos_x + text_width >= image_shape[1] or
                pos_y + text_height >= image_shape[0]):
                continue

            # Рисуем фон для номера с настраиваемой рамкой
            padding = max(2, font_size // 8)
            border_width = max(1, self.contour_thickness // 2)  # Толщина рамки пропорциональна контурам
            draw.rectangle([
                pos_x - padding, pos_y - padding,
                pos_x + text_width + padding, pos_y + text_height + padding
            ], fill=(255, 255, 255), outline=(0, 0, 0), width=border_width)

            # Рисуем номер
            draw.text((pos_x, pos_y), text, fill=(0, 0, 0), font=font)

            # Добавляем в список занятых областей
            occupied_areas.append(text_rect)

        return np.array(contour_template_pil)

    def add_numbers_to_image_optimized(self, image, regions):
        """Добавление номеров красок с умным позиционированием"""
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Список занятых областей
        occupied_areas = []

        # Сортируем регионы по площади (большие области первыми)
        sorted_regions = sorted(regions.items(), key=lambda x: x[1]['area'], reverse=True)

        # Адаптивный порог минимальной площади
        total_image_area = image.shape[0] * image.shape[1]
        min_area_threshold = max(150, total_image_area // 8000)

        for region_id, region_data in sorted_regions:
            color_id = region_data['color_id']
            area = region_data['area']

            # Пропускаем слишком маленькие области
            if area < min_area_threshold:
                continue

            # Адаптивный размер шрифта
            font_size = self.calculate_adaptive_font_size(area, image.shape)

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Определяем цвет текста (контрастный к фону)
            region_color = region_data['color']
            text_color = (0, 0, 0) if np.mean(region_color) > 127 else (255, 255, 255)
            bg_color = (255, 255, 255, 200) if np.mean(region_color) <= 127 else (0, 0, 0, 200)

            # Текст номера краски
            text = str(color_id + 1)

            # Размер текста
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Находим лучшую позицию
            pos_x, pos_y, text_rect = self.find_best_label_position(
                region_data, (text_width, text_height), image.shape, occupied_areas
            )

            # Проверяем границы изображения
            if (pos_x < 0 or pos_y < 0 or
                pos_x + text_width >= image.shape[1] or
                pos_y + text_height >= image.shape[0]):
                continue

            # Рисуем фон для номера
            padding = max(2, font_size // 8)
            draw.rectangle([
                pos_x - padding, pos_y - padding,
                pos_x + text_width + padding, pos_y + text_height + padding
            ], fill=bg_color)

            # Рисуем текст
            draw.text((pos_x, pos_y), text, fill=text_color, font=font)

            # Добавляем в список занятых областей
            occupied_areas.append(text_rect)

        return np.array(pil_image)

    def generate_paint_by_numbers(self, image_path, output_dir="output"):
        """Основной метод генерации картины по номерам"""

        # Создаем директорию для вывода
        os.makedirs(output_dir, exist_ok=True)

        print("Предварительная обработка изображения...")
        processed_image = self.preprocess_image(image_path)

        print("Кластеризация цветов...")
        labels, palette = self.cluster_colors(processed_image)

        print("Создание областей...")
        regions, region_labels = self.create_regions(labels)

        print("Создание контуров...")
        contours = self.create_contours_optimized(regions, processed_image.shape)

        # Создаем кластеризованное изображение более эффективно
        print("Создание кластеризованного изображения...")
        clustered_image = self.create_clustered_image(regions, labels, processed_image.shape)

        print("Создание шаблона для раскрашивания...")
        contour_template = self.create_contour_template(regions, processed_image.shape)

        print("Добавление номеров...")
        final_image = self.add_numbers_to_image_optimized(clustered_image, regions)

        # Комбинируем с контурами
        contour_overlay = cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB)
        final_with_contours = cv2.addWeighted(final_image, 0.8, 255 - contour_overlay, 0.2, 0)

        # Увеличиваем все изображения до оригинального размера
        print("Масштабирование до оригинального размера...")
        original_image = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        clustered_upscaled = self.upscale_to_original_size(clustered_image, self.original_shape)
        contours_upscaled = self.upscale_to_original_size(contours, self.original_shape)
        contour_template_upscaled = self.upscale_pil_image(contour_template, self.original_shape)
        final_upscaled = self.upscale_pil_image(final_image, self.original_shape)

        # Пересчитываем финальное изображение с контурами для оригинального размера
        contour_overlay_upscaled = cv2.cvtColor(contours_upscaled, cv2.COLOR_GRAY2RGB)
        final_with_contours_upscaled = cv2.addWeighted(final_upscaled, 0.8, 255 - contour_overlay_upscaled, 0.2, 0)

        # Сохраняем результаты
        results = {
            'original': original_rgb,
            'clustered': clustered_upscaled,
            'contours': contours_upscaled,
            'contour_template': contour_template_upscaled,
            'final': final_with_contours_upscaled,
            'palette': palette,
            'regions': regions
        }

        self.save_results(results, output_dir)
        self.create_color_palette_guide(palette, output_dir)

        print(f"Готово! Результаты сохранены в {output_dir}")
        return results

    def save_results(self, results, output_dir):
        """Сохранение результатов в оригинальном размере"""
        # Сохраняем основные изображения (все в оригинальном размере)
        cv2.imwrite(f"{output_dir}/01_original.jpg",
                   cv2.cvtColor(results['original'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/02_clustered.jpg",
                   cv2.cvtColor(results['clustered'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/03_contours.jpg", results['contours'])
        cv2.imwrite(f"{output_dir}/04_contour_template.jpg",
                   cv2.cvtColor(results['contour_template'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/05_paint_by_numbers.jpg",
                   cv2.cvtColor(results['final'], cv2.COLOR_RGB2BGR))

        print(f"Все изображения сохранены в размере: {results['original'].shape[:2]} пикселей")

    def create_color_palette_guide(self, palette, output_dir):
        """Создание руководства по цветовой палитре с color_id"""
        # Создаем детальную палитру с подписями
        colors_per_row = min(6, len(palette))  # Максимум 6 цветов в ряду
        rows = (len(palette) + colors_per_row - 1) // colors_per_row

        fig, axes = plt.subplots(rows, colors_per_row, figsize=(colors_per_row * 3, rows * 4))

        # Обрабатываем случай одного ряда
        if rows == 1:
            axes = np.array([axes]) if colors_per_row > 1 else np.array([[axes]])
        elif colors_per_row == 1:
            axes = axes.reshape(-1, 1)

        # Заполняем палитру
        for i, color in enumerate(palette):
            row = i // colors_per_row
            col = i % colors_per_row

            # Нормализуем цвет для отображения
            color_normalized = color / 255.0

            # Создаем превью цвета
            axes[row, col].imshow([[color_normalized]], aspect='auto')

            # Добавляем подробную информацию
            color_id = i + 1
            title_text = f'Color ID: {color_id}\nRGB: ({color[0]}, {color[1]}, {color[2]})\nHEX: #{color[0]:02X}{color[1]:02X}{color[2]:02X}'

            axes[row, col].set_title(title_text, fontsize=10, pad=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

            # Рамка вокруг цвета
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        # Скрываем пустые ячейки
        for i in range(len(palette), rows * colors_per_row):
            row = i // colors_per_row
            col = i % colors_per_row
            axes[row, col].set_visible(False)

        plt.suptitle(f'Цветовая палитра - {len(palette)} красок', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_color_palette.jpg", dpi=300, bbox_inches='tight')
        plt.close()

        # Создаем также текстовый файл с палитрой
        self.save_palette_text(palette, output_dir)

    def save_palette_text(self, palette, output_dir):
        """Сохранение палитры в текстовом виде"""
        with open(f"{output_dir}/color_palette.txt", "w", encoding='utf-8') as f:
            f.write("ПАЛИТРА КРАСОК ДЛЯ КАРТИНЫ ПО НОМЕРАМ\n")
            f.write("="*50 + "\n\n")

            for i, color in enumerate(palette):
                color_id = i + 1
                r, g, b = color[0], color[1], color[2]
                hex_color = f"#{r:02X}{g:02X}{b:02X}"

                f.write(f"Краска №{color_id:2d}:\n")
                f.write(f"  RGB: ({r:3d}, {g:3d}, {b:3d})\n")
                f.write(f"  HEX: {hex_color}\n")
                f.write(f"  Описание: {'Светлый' if np.mean(color) > 127 else 'Темный'} оттенок\n")
                f.write("-" * 30 + "\n")

            f.write(f"\nВсего красок: {len(palette)}\n")
            f.write(f"Дата создания: {np.datetime64('today')}\n")

    def visualize_results(self, results):
        """Визуализация результатов в оригинальном размере"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(results['original'])
        axes[0, 0].set_title(f'Оригинальное изображение\n{results["original"].shape[1]}x{results["original"].shape[0]}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(results['clustered'])
        axes[0, 1].set_title(f'Кластеризованное изображение\n{results["clustered"].shape[1]}x{results["clustered"].shape[0]}')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(results['contours'], cmap='gray')
        axes[0, 2].set_title(f'Контуры областей\n{results["contours"].shape[1]}x{results["contours"].shape[0]}')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(results['contour_template'])
        axes[1, 0].set_title(f'Шаблон для раскрашивания\n{results["contour_template"].shape[1]}x{results["contour_template"].shape[0]}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(results['final'])
        axes[1, 1].set_title(f'Картина по номерам\n{results["final"].shape[1]}x{results["final"].shape[0]}')
        axes[1, 1].axis('off')

        # Показываем палитру цветов
        palette_grid = np.array([results['palette']])
        axes[1, 2].imshow(palette_grid)
        axes[1, 2].set_title(f'Цветовая палитра\n{len(results["palette"])} красок')
        axes[1, 2].axis('off')

        plt.suptitle('Все изображения в оригинальном размере', fontsize=16)
        plt.tight_layout()
        plt.show()


def main():
    """Пример использования"""
    # Создаем генератор с настраиваемой толщиной контуров
    generator = PaintByNumbersGenerator(
        n_colors=16,            # Количество красок
        blur_value=2,           # Сила размытия
        quantization_factor=8,  # Фактор квантизации
        contour_thickness=1     # Толщина контуров (1-10)
    )

    # Путь к изображению (замените на свой)
    image_path = "your_image.jpg"

    try:
        # Генерируем картину по номерам
        results = generator.generate_paint_by_numbers(image_path)

        # Показываем результаты
        generator.visualize_results(results)

        print(f"\nИспользовано {len(results['palette'])} цветов:")
        for i, color in enumerate(results['palette']):
            print(f"  Краска #{i+1}: RGB({color[0]}, {color[1]}, {color[2]})")

        print(f"Толщина контуров: {generator.contour_thickness} пикселей")

    except FileNotFoundError:
        print(f"Файл {image_path} не найден. Укажите правильный путь к изображению.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
