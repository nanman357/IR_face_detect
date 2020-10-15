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

def masking(roi, boundry):
    for (lower, upper) in boundry:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(roi, lower, upper)
        output = cv2.bitwise_and(roi, roi, mask=mask)
        #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        #cv2.imshow("images", np.hstack([roi_bb_rgb, output]))
        (red_thermal_output, green_thermal_output, blue_thermal_output) = cv2.split(output)
        return  np.max(red_thermal_output)
        #print (np.max(red_thermal_output), np.max(green_thermal_output), np.max(blue_thermal_output))


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
    rgb_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2RGB)
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
        roi_bb_rgb = rgb_thermal[int(roi_bb[1]):int(roi_bb[1] + roi_bb[3]), int(roi_bb[0]):int(roi_bb[0] + roi_bb[2])]
        roi_bb_grayscale = grayscale_thermal[int(roi_bb[1]):int(roi_bb[1] + roi_bb[3]), int(roi_bb[0]):int(roi_bb[0] + roi_bb[2])][0]
        cv2.destroyAllWindows()

    #print(roi_bb[1],roi_bb[3],roi_bb[0],roi_bb[2])

    #(minVal_bb, maxVal_bb, minLoc_bb, maxLoc_bb) = cv2.minMaxLoc(roi_bb_rgb)

    #print(minVal_bb, maxVal_bb, minLoc_bb, maxLoc_bb)
    boundry = [([230,0,0],[255,40,40])]



    for(x,y,w,h) in faces:
        #roi_gray = grayscale_color[y:y+h, x:x+w]
        #roi_color = frame_color[y:y+h, x:x+w]

        roi_thermal_face = rgb_thermal[y - color_height_crop: y + h - color_height_crop + 250,
                                        x - color_width_crop : x + w - color_width_crop]




        #(red_thermal_face, green_thermal_face, blue_thermal_face) = cv2.split(roi_thermal_face)
        #(red_bb, green_bb, blue_bb) = cv2.split(roi_bb_rgb)
        #(minVal_thermal, maxVal_thermal, minLoc_thermal, maxLoc_thermal) = cv2.minMaxLoc(roi_thermal_face)
        #(minVal_bb, maxVal_bb, minLoc_bb, maxLoc_bb) = cv2.minMaxLoc(roi_bb_rgb)
        #print(roi_bb_)
        #print(roi_thermal)
        #maxVal_bb = np.max(roi_bb)
        #maxVal_thermal = np.max(roi_thermal)
        #print(height, width, h, w, maxLoc, maxVal)
        #face_frame_count = face_frame_count + 1



        # print('bb: ', np.max(red_bb), np.max(green_bb), np.max(blue_bb))
        # print('bb mean: ', np.mean(red_bb), np.mean(green_bb), np.mean(blue_bb))
        # print('thermal: ', np.max(red_thermal_face), np.max(green_thermal_face), np.max(blue_thermal_face))
        # print('thermal mean: ', np.mean(red_thermal_face), np.mean(green_thermal_face), np.mean(blue_thermal_face))

        # if (cv2.inRange(roi_bb_rgb, np.array[200,0,0], np.array[255,40,40])) and (cv2.inRange(roi_thermal_face, np.array[200,0,0], np.array[255,40,40])):
        #     test = (0, 255, 0)
        # else: #face hotter than bb
        #     test = (255, 0, 0)
        try:
            if masking(roi_thermal_face, boundry) < masking(roi_bb_rgb, boundry):
                test = (0, 255, 0)
            else: #face hotter than bb
                test = (255, 0, 100)
        except:
            continue
        #
        cv2.rectangle(frame_thermal,
                      (x-color_width_crop,y-color_height_crop),
                      (x+w-color_width_crop,y+h-color_height_crop),
                      test,3)
        # cv2.circle(frame_thermal,
        #            (maxLoc_thermal[0] + x - color_width_crop,
        #             maxLoc_thermal[1] + y - color_height_crop),
        #            5, (0, 0, 255), 1)

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