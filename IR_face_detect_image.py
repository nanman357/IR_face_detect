import cv2
import glob

path_img = 'C:\\Users\\Dave\\Downloads\\tufts-face-database-thermal-td-ir\\'
file_imgs = [f for f in glob.glob(path_img + 'TD_IR_E**\\' + '**/*.jpg', recursive=True)]
path_cascades = 'data\\haarcascades\\'
files_cascades = [c for c in glob.glob(path_cascades + '**')]

# for c in files_cascades:
face_cascade = cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')
positive_face_id = 0

for f in file_imgs:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        positive_face_id = positive_face_id + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey()

sensitivity = (positive_face_id / len(file_imgs))
print(c + ': ' + str(sensitivity))

# RESULT
# data\haarcascades\haarcascade_frontalface_alt.xml: 0.5813953488372093
# data\haarcascades\haarcascade_frontalface_alt2.xml: 0.6386404293381037
# data\haarcascades\haarcascade_frontalface_alt_tree.xml: 0.0017889087656529517
# data\haarcascades\haarcascade_frontalface_default.xml: 0.8425760286225402

