import cv2
import numpy

frame = cv2.imread (r'/home/traktirshik/graphic/sylvana.jpg')
#frame = cv2.imread (r'/home/traktirshik/graphic/sylvana.jpg',cv2.IMREAD_GRAYSCALE)
#frame = cv2.imread (r'/home/traktirshik/graphic/sylvana.jpg',cv2.IMREAD_LOAD_GDAL)

cv2.namedWindow ('Display window', cv2.WINDOW_FREERATIO)
#cv2.rectangle(frame,(310,230),(330,190),(0,0,255),2)
#cv2.rectangle(frame,(310,290),(330,250),(0,0,255),2)
#cv2.rectangle(frame,(280,250),(360,230),(0,0,255),2)
dst = cv2.GaussianBlur(frame,(19,19),cv2.BORDER_DEFAULT)

#cv2.namedWindow ('Display window', cv2.WINDOW_NORMAL)
#cv2.namedWindow ('Display window', cv2.WINDOW_FULLSCREEN)


cv2.imshow('Display window',numpy.hstack((frame, dst)))
cv2.waitKey(0)
cv2.destroyAllWindows()