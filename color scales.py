import cv2
import imutils
from threading import Thread
import numpy as np
import matplotlib as plt


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


thermal_video = WebcamVideoStream(src=thermal_url).start()
frame_thermal = thermal_video.read()

frame_thermal = imutils.resize(frame_thermal, width=640)

hsv = cv2.cvtColor(frame_thermal, cv2.COLOR_RGB2HSV)

#cv2.imshow('r',hsv)
cv2.imshow('t', frame_thermal)


h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
# #f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20,10))
# #ax1.set_title('Hue')
# cv2.imshow('a',h)
# #ax2.set_title('Saturation')
# cv2.imshow('b',s)
# #ax3.set_title('Value')
# cv2.imshow('c',v)