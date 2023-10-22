import cv2


frame = cv2.imread (r'/home/traktirshik/graphic/first_lab/sylvana.jpg')


cv2.namedWindow ('Display window HCV', cv2.WINDOW_FREERATIO)
cv2.namedWindow ('Display window no HCV', cv2.WINDOW_FREERATIO)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow('Display window no HCV', frame)


cv2.imshow('Display window HCV', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# H = arccos((0.5 * ((R - G) + (R - B))) / sqrt((R - G)^2 + (R - B) * (G - B))) * (180 / π)
# C = 100 * (1 - min(R, G, B) / V)
# V = max(R, G, B)
