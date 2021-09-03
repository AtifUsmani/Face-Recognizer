import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('frontal_face_default.xml')

webcam = cv2.VideoCapture(0) #If you have multiple webcams , change the number in the brackets.

while True:
    successful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    print(face_coordinates)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128 ,256)),4)
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    if key==27:
        break
        #press esc to exit the program

webcam.release()