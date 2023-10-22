import cv2
import numpy as np
import time

# Открываем видеопоток с веб-камеры
cap = cv2.VideoCapture(0)

# Устанавливаем разрешение видео
cap.set(3, 320)
cap.set(4, 240)

# Создаем окно настроек
cv2.namedWindow("Control")

# Начальные значения параметров для цветового фильтра в HSV
iLowH = 0
iHighH = 25
iLowS = 100
iHighS = 255
iLowV = 140
iHighV = 255

# Создаем ползунки для настройки параметров цветового фильтра
cv2.createTrackbar("LowH", "Control", iLowH, 179, lambda x: None)  # Добавлен пустой обработчик
cv2.createTrackbar("HighH", "Control", iHighH, 179, lambda x: None)  # Добавлен пустой обработчик
cv2.createTrackbar("LowS", "Control", iLowS, 255, lambda x: None)  # Добавлен пустой обработчик
cv2.createTrackbar("HighS", "Control", iHighS, 255, lambda x: None)  # Добавлен пустой обработчик
cv2.createTrackbar("LowV", "Control", iLowV, 255, lambda x: None)  # Добавлен пустой обработчик
cv2.createTrackbar("HighV", "Control", iHighV, 255, lambda x: None)  # Добавлен пустой обработчик


def update_values(x):
    global iLowH, iHighH, iLowS, iHighS, iLowV, iHighV
    iLowH = cv2.getTrackbarPos("LowH", "Control")
    iHighH = cv2.getTrackbarPos("HighH", "Control")
    iLowS = cv2.getTrackbarPos("LowS", "Control")
    iHighS = cv2.getTrackbarPos("HighS", "Control")
    iLowV = cv2.getTrackbarPos("LowV", "Control")
    iHighV = cv2.getTrackbarPos("HighV", "Control")

# Установите обработчик события для обновления параметров
cv2.createTrackbar("LowH", "Control", iLowH, 179, update_values)
cv2.createTrackbar("HighH", "Control", iHighH, 179, update_values)
cv2.createTrackbar("LowS", "Control", iLowS, 255, update_values)
cv2.createTrackbar("HighS", "Control", iHighS, 255, update_values)
cv2.createTrackbar("LowV", "Control", iLowV, 255, update_values)
cv2.createTrackbar("HighV", "Control", iHighV, 255, update_values)


start_time = time.time()
frames = 0

while True:
    ret, imgOriginal = cap.read()  # Читаем новый кадр

    if not ret:
        print("Cannot read a frame from video stream")
        break

    # Переводим изображение в цветовое пространство HSV
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    # Применяем цветовой фильтр
    lower_bound = np.array([iLowH, iLowS, iLowV])
    upper_bound = np.array([iHighH, iHighS, iHighV])
    imgThresholded = cv2.inRange(imgHSV, lower_bound, upper_bound)

    # Морфологические операции
    kernel = np.ones((5, 5), np.uint8)
    imgThresholded = cv2.erode(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.dilate(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.dilate(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.erode(imgThresholded, kernel, iterations=1)

    # Вычисляем моменты изображения
    moments = cv2.moments(imgThresholded)
    dM01 = moments["m01"]
    dM10 = moments["m10"]
    dArea = moments["m00"]

    if dArea > 10000:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)

        # Рисуем горизонтальную линию (горизонтальная палка креста)
        cv2.line(imgOriginal, (x - 10, y), (x + 10, y), (255, 0, 0), 2)

        # Рисуем вертикальную линию (вертикальная палка креста)
        cv2.line(imgOriginal, (x, y - 10), (x, y + 10), (255, 0, 0), 2)


        posX = int(dM10 / dArea)
        posY = int(dM01 / dArea)

        # Находим ограничивающий прямоугольник вокруг объекта
        x, y, w, h = cv2.boundingRect(imgThresholded)

        # Рисуем прямоугольник
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Отображаем результат
    cv2.imshow("Thresholded Image", imgThresholded)
    cv2.imshow("Original", imgOriginal)
    cv2.imshow("HCV", imgHSV)

    if cv2.waitKey(30) == 27:  # Ждем нажатия клавиши "Esc" для выхода
        print("esc key is pressed by user")
        break

    frames += 1

end_time = time.time()
dif = end_time - start_time
print("FPS: {:.2f}".format(frames / dif))

# Закрываем все окна и освобождаем видеопоток
cap.release()
cv2.destroyAllWindows()
