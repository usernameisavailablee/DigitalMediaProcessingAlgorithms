import cv2
import numpy as np

# Загрузка классов для обнаружения (может потребоваться заменить на другие классы для обнаружения лиц)
with open("darknet/data/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Загрузка модели YOLO
net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")

# Запуск видеопотока с камеры
cap = cv2.VideoCapture(0)  # 0 - основная камера, 1 - внешняя камера (если есть)

while True:
    # Захват видеопотока
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Подготовка входных данных для обработки через YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Установка минимальной уверенности для обнаружения объекта
            if confidence > 0.5 and classes[class_id] == 'person':
                # Получение координат обнаруженного лица
                box = detection[0:4] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype("int")

                # Отрисовка рамки вокруг лица на кадре
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Показ обработанного кадра с обнаруженными лицами
    cv2.imshow("Real-time Face Detection", frame)

    # Остановка выполнения при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
