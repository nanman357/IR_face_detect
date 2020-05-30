import cv2

color_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=1&subtype=1"
thermal_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=2&subtype=1"

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')
color_video = cv2.VideoCapture(color_url)
thermal_video = cv2.VideoCapture(thermal_url)

frame_count = 0
face_frame_count = 0
while True:
    check_color, frame_color = color_video.read()
    check_thermal, frame_thermal = color_video.read()
    grayscale_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    grayscale_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_color,
                                          scaleFactor=1.3,
                                          minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(grayscale_color,
                      (x,y),
                      (x+w,y+h),
                      (255,0,0),1)
        roi_gray = grayscale_color[y:y+h, x:x+w]
        #roi_color = frame_color[y:y+h, x:x+w]
        roi_thermal = grayscale_thermal[y:y+h, x:x+w]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_thermal)
        #print(height, width, h, w, maxLoc, maxVal)
        #face_frame_count = face_frame_count + 1
        cv2.circle(grayscale_color, (maxLoc[0] + x, maxLoc[1] + y), 5, (0, 0, 255), 2)
    cv2.imshow('live_video', grayscale_color)

    key = cv2.waitKey(1)
    frame_count = frame_count + 1

    if key == ord('q'):
        print('frame_count ', frame_count)
        print('face_frame_count ', face_frame_count)
        break


color_video.release()
cv2.destroyAllWindows()