#include <opencv2/opencv.hpp>

// Задание 1: Построение матрицы Гаусса
cv::Mat gaussianMatrix(int size, double sigma) {
    cv::Mat kernel(size, size, CV_64F);
    double sum = 0.0;
    int halfSize = size / 2;

    for (int x = -halfSize; x <= halfSize; x++) {
        for (int y = -halfSize; y <= halfSize; y++) {
            double value = (1 / (2 * M_PI * sigma * sigma)) *
                exp(-((x * x + y * y) / (2 * sigma * sigma)));
            kernel.at<double>(x + halfSize, y + halfSize) = value;
            sum += value;
        }
    }

    // Нормализация
    kernel /= sum;

    return kernel;
}

// Задание 3: Реализовать фильтр Гаусса средствами C++
cv::Mat gaussianBlur(const cv::Mat& inputImage, int size, double sigma) {
    cv::Mat kernel = gaussianMatrix(size, sigma);
    cv::Mat result;
    cv::filter2D(inputImage, result, -1, kernel);
    return result;
}

int main() {
    std::string imagePath = "img.jpg";
    std::string outputPath = "output_image.jpg";

    // Задайте размеры и значение сигмы
    std::vector<int> sizes = {3, 5, 7};
    double sigma = 5.0; // Задайте нужное значение сигмы

    // Проход по различным размерам и применение фильтра Гаусса
    for (int size : sizes) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::Mat blurredImage = gaussianBlur(image, size, sigma);
        cv::imwrite("output_image_size_" + std::to_string(size) + "_sigma_" + std::to_string(sigma) + ".jpg", blurredImage);
    }

    // Применение фильтра Гаусса с использованием OpenCV
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat blurredImageOpencv;
    cv::GaussianBlur(image, blurredImageOpencv, cv::Size(5, 5), 1.0);
    cv::imwrite("output_image_opencv.jpg", blurredImageOpencv);

    std::cout << "All tasks completed." << std::endl;

    return 0;
}
