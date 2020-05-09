import face_recognition
import os
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier("data\\haarcascades\\haarcascade_frontalface_default.xml")
video = cv2.VideoCapture("C:\\Users\\Dave\\Downloads\\ytdl\\1.mp4")
a=1

while True:
    a=a+1
    check, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0),1)
        roi_gray = grayscale[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]

    cv2.imshow('Capturing', grayscale)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows()