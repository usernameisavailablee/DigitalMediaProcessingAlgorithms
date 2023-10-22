import numpy as np
import cv2

# Задание 1: Построение матрицы Гаусса
def build_gaussian_matrix(n, sigma):
    center = n // 2
    gaussian_matrix = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            gaussian_matrix[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return gaussian_matrix

# Задание 2: Нормирование матрицы Гаусса
def normalize_gaussian_matrix(gaussian_matrix):
    return gaussian_matrix / np.sum(gaussian_matrix)

# Задание 3: Реализация фильтра Гаусса
def apply_gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Задание 4: Применение фильтра Гаусса для изображения
def apply_gaussian_blur_to_image(image, sigma, kernel_size):
    gaussian_matrix = build_gaussian_matrix(kernel_size, sigma)
    normalized_gaussian_matrix = normalize_gaussian_matrix(gaussian_matrix)
    blurred_image = apply_gaussian_filter(image, normalized_gaussian_matrix)
    return blurred_image

# Загрузка изображения
image_path = "img.jpg"
image = cv2.imread(image_path)

# Задание 4: Применение фильтра Гаусса для разных значений sigma и размеров матрицы свертки
sigma_1 = 1.0
sigma_2 = 2.0
kernel_size_1 = 3
kernel_size_2 = 5

blurred_image_1 = apply_gaussian_blur_to_image(image, sigma_1, kernel_size_1)
blurred_image_2 = apply_gaussian_blur_to_image(image, sigma_1, kernel_size_2)
blurred_image_3 = apply_gaussian_blur_to_image(image, sigma_2, kernel_size_1)
blurred_image_4 = apply_gaussian_blur_to_image(image, sigma_2, kernel_size_2)

# Задание 5: Размытие Гаусса с использованием OpenCV
blurred_image_opencv = cv2.GaussianBlur(image, (kernel_size_1, kernel_size_1), sigma_1)

# Вывод результатов
cv2.imshow("Original Image", image)
cv2.imshow("Blurred Image 1", blurred_image_1)
cv2.imshow("Blurred Image 2", blurred_image_2)
cv2.imshow("Blurred Image 3", blurred_image_3)
cv2.imshow("Blurred Image 4", blurred_image_4)
cv2.imshow("Blurred Image OpenCV", blurred_image_opencv)
cv2.waitKey(0)
cv2.destroyAllWindows()
