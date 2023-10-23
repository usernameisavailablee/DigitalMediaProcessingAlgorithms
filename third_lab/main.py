from gauss_methods import *
# Выполнение заданий
sizes = [3, 5, 7]
sigma = 5.0  # Задайте нужное значение сигмы

for size in sizes:
    gauss_matrix = gaussian_matrix(size, sigma)
    normalized_matrix = normalize_gaussian_matrix(gauss_matrix)
    print(f"Normalized Gaussian Matrix for size {size}x{size}:\n{normalized_matrix}\n")

image_path = 'input_images/img.jpg'  # Замените на свой путь к изображению
output_path = 'output_images/output_image.jpg'  # Путь для сохранения размытого изображения

for size in sizes:
    output_path_size = f'output_images/output_image_size_{size}_sigma_{sigma}.jpg'
    apply_gaussian_blur(image_path, size, sigma, output_path_size)
    blurred_image = cv2.imread(output_path_size)
    cv2.imshow(f"size {size}, sigma {sigma}", blurred_image)
    cv2.waitKey(0)  # Ожидание нажатия клавиши

output_path_opencv = 'output_images/output_image_opencv.jpg'  # Путь для сохранения размытого изображения с OpenCV
gaussian_blur_opencv(image_path, output_path_opencv, size=5, sigma=1.0)

print("All tasks completed.")
