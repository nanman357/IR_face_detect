import cv2
#url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0" - regular
#url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=2&subtype=1" - thermal
url = 'D:\\thermal\\tufts.mov'

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(url)

frame_count = 0
face_frame_count = 0
while True:
    check, frame = video.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale,
                                          scaleFactor=1.3,
                                          minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(grayscale,
                      (x,y),
                      (x+w,y+h),
                      (255,0,0),1)
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        face_frame_count = face_frame_count + 1
    cv2.imshow('Video_yt', grayscale)

    key = cv2.waitKey(40)
    frame_count = frame_count + 1

    if key == ord('q'):
        print('frame_count ', frame_count)
        print('face_frame_count ', face_frame_count)
        break


video.release()
cv2.destroyAllWindows()