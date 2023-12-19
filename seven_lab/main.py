import pytesseract
import cv2

img = cv2.imread('data/1-.jpg')

text = pytesseract.image_to_string(img,'rus+eng')
print(text)
