import numpy as np
import cv2

# Задание 1: Построение матрицы Гаусса
def gaussian_matrix(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                     np.exp(- ((x - (size-1)/2) ** 2 + (y - (size-1)/2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Задание 2: Нормировать матрицу Гаусса
def normalize_gaussian_matrix(matrix):
    return matrix / np.sum(matrix)

# Задание 3: Реализовать фильтр Гаусса средствами Python
def gaussian_blur(image, size, sigma):
    kernel = gaussian_matrix(size, sigma)
    result = cv2.filter2D(image, -1, kernel)
    return result

# Задание 4: Применить фильтр Гаусса с разными параметрами
def apply_gaussian_blur(image_path, size, sigma, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blurred_image = gaussian_blur(image, size, sigma)
    cv2.imwrite(output_path, blurred_image)

# Задание 5: Реализовать размытие Гаусса с использованием OpenCV
def gaussian_blur_opencv(image_path, output_path, size, sigma):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blurred_image = cv2.GaussianBlur(image, (size, size), sigma)
    cv2.imwrite(output_path, blurred_image)
