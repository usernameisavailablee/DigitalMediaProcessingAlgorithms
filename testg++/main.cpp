#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    // Загрузка изображения с диска
    cv::Mat image = cv::imread("sample.jpg");

    if (!image.data) {
        std::cout << "Не удалось загрузить изображение." << std::endl;
        return -1;
    }

    // Отображение изображения в окне
    cv::imshow("Loaded Image", image);

    // Ожидание нажатия клавиши
    cv::waitKey(0);

    return 0;
}
