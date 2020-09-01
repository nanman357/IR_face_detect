import cv2
import imutils
from threading import Thread


color_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0"
thermal_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=2&subtype=0"

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')
#color_video = cv2.VideoCapture(color_url)
#thermal_video = cv2.VideoCapture(thermal_url)

frame_count = 0
face_frame_count = 0

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame


    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

color_video = WebcamVideoStream(src=color_url).start()
thermal_video = WebcamVideoStream(src=thermal_url).start()

while True:
    frame_color = color_video.read()
    frame_color = imutils.resize(frame_color, width=640)
    frame_thermal = thermal_video.read()
    frame_thermal = imutils.resize(frame_thermal, width=640)

    grayscale_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    grayscale_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_color,
                                          scaleFactor=1.3,
                                          minNeighbors=5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame_color,
                      (x,y),
                      (x+w,y+h),
                      (255,0,0),1)
        roi_gray = grayscale_color[y:y+h, x:x+w]
        #roi_color = frame_color[y:y+h, x:x+w]
        roi_thermal = grayscale_thermal[y:y+h, x:x+w]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_thermal)
        #print(height, width, h, w, maxLoc, maxVal)
        #face_frame_count = face_frame_count + 1
        cv2.circle(frame_color, (maxLoc[0] + x, maxLoc[1] + y), 5, (0, 0, 255), 2)


    cv2.imshow('live_video', frame_color)
    cv2.imshow('thermal_video', frame_thermal)


    cv2.imshow('live_video', frame_color)
    cv2.imshow('thermal_video', frame_thermal)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# while True:
#     check_color, frame_color = color_video.read()
#     check_thermal, frame_thermal = thermal_video.read()
#
#     grayscale_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
#     grayscale_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(grayscale_color,
#                                           scaleFactor=1.3,
#                                           minNeighbors=5)
#
#     for(x,y,w,h) in faces:
#         cv2.rectangle(frame_color,
#                       (x,y),
#                       (x+w,y+h),
#                       (255,0,0),1)
#         roi_gray = grayscale_color[y:y+h, x:x+w]
#         #roi_color = frame_color[y:y+h, x:x+w]
#         roi_thermal = grayscale_thermal[y:y+h, x:x+w]
#         (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_thermal)
#         #print(height, width, h, w, maxLoc, maxVal)
#         #face_frame_count = face_frame_count + 1
#         cv2.circle(frame_color, (maxLoc[0] + x, maxLoc[1] + y), 5, (0, 0, 255), 2)
#     cv2.imshow('live_video', frame_color)
#     cv2.imshow('thermal_video', frame_thermal)
#
#     key = cv2.waitKey(1)
#     frame_count = frame_count + 1
#
#     if key == ord('q'):
#         print('frame_count ', frame_count)
#         print('face_frame_count ', face_frame_count)
#         break

# color_video.release()
# cv2.destroyAllWindows()
#
cv2.destroyAllWindows()
color_video.stop()
thermal_video.stop()