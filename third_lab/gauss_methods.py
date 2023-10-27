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

def apply_gaussian_blur(image,size,sigma):
    blurred_image = gaussian_blur(image,size,sigma)
    return blurred_image
    

# Задание 5: Реализовать размытие Гаусса с использованием OpenCV
def gaussian_blur_opencv(image_path, output_path, size, sigma):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    blurred_image = cv2.GaussianBlur(image, (size, size), sigma)
    cv2.imwrite(output_path, blurred_image)

def my_filter(img, kernel):
    try:
      img_height, img_width, img_canals = img.shape
    except:
      img_height, img_width = img.shape
      img_canals = 1

    kernel_height, kernel_width = kernel.shape

    result_height = img_height - kernel_height + 1
    result_width = img_width - kernel_width + 1

    result = np.zeros((result_height, result_width), dtype=np.float32)

    if img_canals != 1:
      for i in range(result_height):
        for j in range(result_width):
          for canal in range(img_canals):
            result[i, j, canal] = np.sum(img[i:i + kernel_height, j:j + kernel_width, canal] * kernel)
    else:
      for i in range(result_height):
        for j in range(result_width):
          result[i, j] = np.sum(img[i:i + kernel_height, j:j + kernel_width] * kernel)
    return result
