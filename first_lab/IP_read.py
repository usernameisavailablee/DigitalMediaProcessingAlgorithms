import cv2


url = 'http://192.168.0.104:8080/video'

cap = cv2.VideoCapture(url)


if not cap.isOpened():
    print("Ошибка: Не удается подключиться к камере")
    exit()

# Чтение и отображение видеопотока
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera Feed', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
