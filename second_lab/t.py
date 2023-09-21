import cv2
import numpy as np

def readWriteTOFile():

    video = cv2.VideoCapture(0)
    ok, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))


    while True:
        ok, frame = video.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Задание 2: Применяем фильтрацию для красной части
        lower_red = np.array([0, 100, 100])  # Нижний порог для красной компоненты
        upper_red = np.array([10, 255, 255])  # Верхний порог для красной компоненты

        mask = cv2.inRange(hsv, lower_red, upper_red)  # Создаем маску для красной части

        # Отображаем только красную часть изображения
        red_only = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', red_only)
        video_writer.write(red_only)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

readWriteTOFile()
