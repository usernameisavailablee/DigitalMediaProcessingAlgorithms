import sys
import os

current_dir = os.path.dirname(__file__)

third_lab_path = os.path.join(current_dir, "..", "third_lab")
sys.path.append(third_lab_path)

from gauss_methods import *

class CannyEdgeDetector:
    def start(self, path, filter_shape, sigma, low_pr, high_pr):
        img = self.prepare_img(path, filter_shape, sigma)
        gradient_magnitude, gradient_angle = self.search_for_gradients(img)
        suppressed = self.non_max_suppression(gradient_magnitude, gradient_angle)
        result_of_method = self.double_threshold_filtering(suppressed, low_pr, high_pr)

        return result_of_method

    def prepare_img(self, path, filter_shape, sigma):
        try:
            img = cv2.imread(path)
        except:
            img = cv2.imread('Ленок.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return self.apply_gaussian_blur(img, filter_shape, sigma, "output_image.jpg")

    def apply_gaussian_blur(self, img, filter_shape, sigma, output_path):
        kernel = cv2.getGaussianKernel(filter_shape, sigma)
        kernel = kernel * kernel.T
        img = cv2.filter2D(img, -1, kernel)

        return img

    def get_sobel_operator_x(self):
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    def get_sobel_operator_y(self):
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    def search_for_gradients(self, img):
        gradient_x = my_filter(img, self.get_sobel_operator_x())
        gradient_y = my_filter(img, self.get_sobel_operator_y())

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_angle = np.arctan2(gradient_y, gradient_x)
        gradient_angle = np.round(gradient_angle / (np.pi / 4)) * (np.pi / 4)

        return gradient_magnitude, gradient_angle

    def non_max_suppression(self, gradient_magnitude, gradient_angle):
        height, width = gradient_magnitude.shape
        suppressed = np.copy(gradient_magnitude)

        gradient_angle = (gradient_angle * 180.0 / np.pi) % 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                angle = gradient_angle[i, j]

                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
                elif (22.5 <= angle < 67.5):
                    neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
                elif (67.5 <= angle < 112.5):
                    neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

                if gradient_magnitude[i, j] < max(neighbors):
                    suppressed[i, j] = 0

        return suppressed

    def double_threshold_filtering(self, img, low_pr, high_pr):
        down = low_pr * 255
        up = high_pr * 255

        n, m = img.shape
        clone_of_img = np.copy(img)
        for i in range(n):
            for j in range(m):
                if clone_of_img[i, j] >= up:
                    clone_of_img[i, j] = 255
                elif clone_of_img[i, j] <= down:
                    clone_of_img[i, j] = 0
                else:
                    clone_of_img[i, j] = 127

        return clone_of_img

# Пример использования:
detector = CannyEdgeDetector()
result = detector.start("img_djaina.jpg", 7, 1.0, 0.5, 1)
cv2.imwrite("canny_output_dj.jpg", result)
