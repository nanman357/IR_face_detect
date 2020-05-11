import cv2
from os import listdir
from os.path import isfile, join
import glob

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')

path = 'C:\\Users\\Dave\\Downloads\\tufts-face-database-thermal-td-ir\\'
files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]
face_count = 0

for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face_count = face_count + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #cv2.imshow('img', img)
    cv2.waitKey()
    #print(face_count)

print('sensitivity: ' + str(face_count/len(files)))