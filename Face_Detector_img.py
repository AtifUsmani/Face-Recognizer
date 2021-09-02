import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('frontal_face_default.xml')

img = cv2.imread('Face.jpg')

grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
print(face_coordinates)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128 ,256)),4)

cv2.imshow('Face Detector', img)
key = cv2.waitKey()

print("Code Completed")