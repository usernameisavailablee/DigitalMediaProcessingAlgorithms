import cv2

cap = cv2.VideoCapture(r'/home/traktirshik/graphic/sylvana.mp4',cv2.CAP_ANY)

#cv2.namedWindow ('Display window', cv2.WINDOW_FREERATIO)

while (True):
	ret, frame = cap.read()

	if not(ret):
		break

	frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', frame)
	if cv2.waitKey(100) & 0xFF == 27:
		break