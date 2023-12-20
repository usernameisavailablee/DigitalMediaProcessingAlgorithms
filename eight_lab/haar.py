import cv2

# Загрузка предварительно обученного классификатора для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Запуск видеопотока с камеры
cap = cv2.VideoCapture(0)  # 0 - основная камера, 1 - внешняя камера (если есть)

while True:
    # Захват видеопотока
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Показ обработанного кадра с обнаруженными лицами
    cv2.imshow("Real-time Face Detection", frame)

    # Остановка выполнения при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()


# import cv2
# import time
#
# # Загрузка предварительно обученного классификатора для лиц
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # Путь к изображению для обработки
# image_path = '/home/traktirshik/graphic/eight_lab/test/yolov8-face-landmarks-opencv-dnn/images/1.jpg'
#
# # Чтение изображения
# frame = cv2.imread(image_path)
#
# start_time = time.time()  # Записываем текущее время до обработки кадра
#
# # Преобразование кадра в оттенки серого
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# # Обнаружение лиц
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
# # Отрисовка прямоугольников вокруг обнаруженных лиц
# for (x, y, w, h) in faces:
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# end_time = time.time()  # Записываем текущее время после обработки кадра
# elapsed_time = end_time - start_time  # Вычисляем затраченное время
#
# print(f"Затраченное время на обработку кадра: {elapsed_time} секунд")
# num_detected_faces = len(faces)
# print(f"Количество обнаруженных лиц: {num_detected_faces}")
#
# cv2.namedWindow("Face Detection in Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Face Detection in Image", 640, 720)
# # Показ обработанного кадра с обнаруженными лицами
# cv2.imshow("Face Detection in Image", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
