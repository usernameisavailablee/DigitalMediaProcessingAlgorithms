import cv2
import numpy as np

def blur_rectangle(frame, x, y, width, height, blur_kernel_size=(15, 15)):
    # Создаем копию фрейма, чтобы не изменять оригинальный кадр.
    blurred_frame = frame.copy()

    # Определяем координаты верхнего левого и нижнего правого углов прямоугольника.
    top_left = (x, y)
    bottom_right = (x + width, y + height)

    # Выделяем прямоугольник на копии фрейма.
    roi = blurred_frame[y:y+height, x:x+width]

    # Применяем GaussianBlur к выделенному прямоугольнику.
    blurred_roi = cv2.GaussianBlur(roi, blur_kernel_size, 0)

    # Заменяем выделенную область на фрейме на размытую версию.
    blurred_frame[y:y+height, x:x+width] = blurred_roi

    return blurred_frame



def readIPWriteTOFile():
    video = cv2.VideoCapture(0)
    ok, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))

    width_rectangle_h = 80
    height_rectangle_h = 20

    width_rectangle_v = 100
    height_rectangle_v = 20

    x = int (((w - width_rectangle_h)/2))
    y = int(((h - height_rectangle_h)/2))

    top_left_for_h = (x,y)
    bottom_right_for_h = (x+width_rectangle_h, y + height_rectangle_h)

    x1 = int (((h - width_rectangle_v)/2))
    y1 = int (((w - height_rectangle_v)/2))

    top_left_for_v = (y1,x1)
    bottom_right_for_v = (y1+ height_rectangle_v, x1 +  int((width_rectangle_v - height_rectangle_h) / 2))

    x2 = int (((h - width_rectangle_v )/2) + width_rectangle_v/2 - height_rectangle_h/2 + height_rectangle_h)
    y2 = int (((w - height_rectangle_v)/2))

    top_left_for_v1 = (y2,x2)
    bottom_right_for_v1 = (y2+ height_rectangle_v, x2 +  int((width_rectangle_v - height_rectangle_h) / 2))


#zxc
    while True:
        ok, frame = video.read()


        cv2.rectangle(frame, top_left_for_v, bottom_right_for_v, (0, 0, 255), 2)
        cv2.rectangle(frame, top_left_for_v1, bottom_right_for_v1, (0, 0, 255), 2)
        cv2.rectangle(frame, top_left_for_h, bottom_right_for_h, (0, 0, 255), 2)

        blurred_frame = blur_rectangle(frame, x, y, width_rectangle_h, height_rectangle_h)


        cv2.imshow('frame', blurred_frame)

        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Ваша функция blur_rectangle должна быть определена в коде до вызова readIPWriteTOFile


def print_cam():
    # Создаем объект VideoCapture для чтения с камеры.
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры по умолчанию

    # Устанавливаем размеры кадра (ширина и высота) на 640x480.
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        # Захватываем кадр с камеры.
        ret, frame = cap.read()



        # Преобразуем кадр в черно-белый.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Отображаем кадр в окне с именем "frame".

        cv2.imshow('frame', frame)

        if not ret:
            print("Ошибка при захвате кадра.")
            break
        # Если нажата клавиша 'Esc' (код 27 в ASCII), то выходим из цикла.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Когда всё закончено, освобождаем ресурсы и закрываем окно.
    cap.release()
    cv2.destroyAllWindows()

# Вызываем функцию для запуска камеры.
#print_cam()

readIPWriteTOFile()
