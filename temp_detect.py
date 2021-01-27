import cv2
import glob
import numpy as np

path_img = 'D:\\thermal\\tufts-face-database-thermal-td-ir\\'
file_imgs = [f for f in glob.glob(path_img + 'TD_IR_E**\\' + '**/*.jpg', recursive=True)]
path_cascades = 'data\\haarcascades\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path_cascades)
positive_face_id = 0
print(len(file_imgs))
for f in file_imgs:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        positive_face_id = positive_face_id + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[y:y+h, x:x+w]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi)
        print(height, width, h, w, maxLoc, maxVal)
        cv2.circle(img, (maxLoc[0]+x, maxLoc[1]+y), 5, (0, 0, 255), 2)
        #cv2.circle(roi, (maxLoc[0], maxLoc[1]), 5, (0, 0, 255), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#sensitivity = (positive_face_id / len(file_imgs))
#print(c + ': ' + str(sensitivity))