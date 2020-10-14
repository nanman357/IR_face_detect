import cv2
import imutils
from threading import Thread
import numpy as np


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

# width_color = frame_color.shape[0]
# height_color = frame_color.shape[1]
# width_thermal = frame_thermal.shape[0]
# heigth_thermal = frame_thermal.shape[1]

#print(width_color, height_color)
#print(width_thermal, heigth_thermal)

color_width_original = 1920
color_height_origial = 1080
color_width_crop = 320
color_height_crop = 60

roi_bb = None
while True:
    frame_color = color_video.read()
    frame_thermal = thermal_video.read()
    #print(frame_thermal.dtype) #bitrate
    #print('thermal', frame_thermal.shape)
    #print('color', frame_color.shape)
    #grayscale_color = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    grayscale_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_color,
                                          scaleFactor=1.3,
                                          minNeighbors=5)


    # crop color frame to size roughly of thermal frame
    frame_color = frame_color[color_height_crop:color_height_origial - color_height_crop,
                              color_width_crop:color_width_original - color_width_crop]

    #select area of black-body on thermal camera
    while roi_bb == None:
        roi_bb = cv2.selectROI(frame_thermal)
        roi_bb_ = grayscale_thermal[int(roi_bb[1]):int(roi_bb[1]+roi_bb[3]), int(roi_bb[0]):int(roi_bb[0]+roi_bb[2])]
        cv2.destroyAllWindows()
        roi_bb_ = roi_bb_[0]
    #print(roi_bb[1],roi_bb[3],roi_bb[0],roi_bb[2])
    #print(roi_bb_[0])

    for(x,y,w,h) in faces:
        #roi_gray = grayscale_color[y:y+h, x:x+w]
        #roi_color = frame_color[y:y+h, x:x+w]

        roi_thermal = grayscale_thermal[y - color_height_crop : y + h - color_height_crop,
                                        x - color_width_crop : x + w - color_width_crop][0]
        (minVal_thermal, maxVal_thermal, minLoc_thermal, maxLoc_thermal) = cv2.minMaxLoc(roi_thermal)
        (minVal_bb, maxVal_bb, minLoc_bb, maxLoc_bb) = cv2.minMaxLoc(roi_bb_)
        print(roi_bb_)
        print(roi_thermal)
        #maxVal_bb = np.max(roi_bb)
        #maxVal_thermal = np.max(roi_thermal)
        #print(height, width, h, w, maxLoc, maxVal)
        #face_frame_count = face_frame_count + 1
        print('bb: ', maxVal_bb)
        print('thermal: ', maxVal_thermal)
        if maxVal_bb >= maxVal_thermal:
            test = (255,0,0)
        else:
            test = (0,0,255)

        cv2.rectangle(frame_thermal,
                      (x-color_width_crop,y-color_height_crop),
                      (x+w-color_width_crop,y+h-color_height_crop),
                      test,1)
        cv2.circle(frame_thermal,
                   (maxLoc_thermal[0] + x - color_width_crop,
                    maxLoc_thermal[1] + y - color_height_crop),
                   5, (0, 0, 255), 1)

    frame_thermal = imutils.resize(frame_thermal, width=640)
    frame_color = imutils.resize(frame_color, width=640)

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