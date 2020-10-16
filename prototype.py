import cv2
import imutils
from threading import Thread
import numpy as np

color_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=1&subtype=0"
thermal_url = "rtsp://admin:admin@192.168.1.108:80/cam/realmonitor?channel=2&subtype=0"

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')
# color_video = cv2.VideoCapture(color_url)
# thermal_video = cv2.VideoCapture(thermal_url)

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


def masking(roi, bounds):
    for (lower, upper) in bounds:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        try:
            mask = cv2.inRange(roi, lower, upper)
            output = cv2.bitwise_and(roi, roi, mask=mask)
            (red_thermal_output, green_thermal_output, blue_thermal_output) = cv2.split(output)
            return np.max(red_thermal_output)
        except:
            break


def minmaxlox(input, roi):
    grayscale_thermal = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    roi_grayscale = grayscale_thermal[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi_grayscale)
    return minVal, maxVal, minLoc, maxLoc

color_video = WebcamVideoStream(src=color_url).start()
thermal_video = WebcamVideoStream(src=thermal_url).start()

color_width_original = 1920
color_height_origial = 1080
color_width_crop = 320
color_height_crop = 60

boundary = [([230, 0, 0], [255, 40, 40])]
roi_bb = None

while True:
    frame_color = color_video.read()
    frame_thermal = thermal_video.read()
    rgb_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(frame_color,
                                          scaleFactor=1.3,
                                          minNeighbors=5)

    # crop color frame to size roughly of thermal frame
    frame_color = frame_color[color_height_crop:color_height_origial - color_height_crop,
                  color_width_crop:color_width_original - color_width_crop]

    # select area of black-body on thermal camera
    while roi_bb == None:
        roi_bb = cv2.selectROI(frame_thermal)
        roi_bb_rgb = rgb_thermal[int(roi_bb[1]):int(roi_bb[1] + roi_bb[3]), int(roi_bb[0]):int(roi_bb[0] + roi_bb[2])]
        cv2.destroyAllWindows()

    throat_correction = 250 # used to include chin and throat in roi_thermal_face
    # face detection for-loop & selecting faces with temp > bb
    for (x, y, w, h) in faces:
        roi_thermal_face = rgb_thermal[y - color_height_crop: y + h - color_height_crop + throat_correction,
                           x - color_width_crop: x + w - color_width_crop]

        roi_thermal_face_result = masking(roi_thermal_face, boundary)
        roi_bb_rgb_result = masking(roi_bb_rgb, boundary)

        try:
            if roi_thermal_face_result < roi_bb_rgb_result:
                # bb is hottest & has 'red mark'
                test = (0, 255, 0)
                print('bb')
            elif roi_thermal_face_result > roi_bb_rgb_result:
                # face is hottest and has 'red mark'
                test = (0, 0, 255)
                print('face')
            elif roi_thermal_face_result == roi_bb_rgb_result and roi_thermal_face_result == 0:
                # no red mark in roi's
                roi_thermal_face_2 = (x - color_width_crop, y - color_height_crop, w, h + throat_correction)
                if minmaxlox(frame_thermal, roi_bb)[0] < minmaxlox(frame_thermal, roi_thermal_face_2)[0]:
                    # bb is hotter but has no 'red mark'
                    print(minmaxlox(frame_thermal, roi_bb)[0])
                    print(minmaxlox(frame_thermal, roi_thermal_face_2)[0])
                    test = (0, 255, 0)
                    print('bb other')
                else:
                    # face is hotter but has no 'red mark'
                    print(minmaxlox(frame_thermal, roi_bb)[0])
                    print(minmaxlox(frame_thermal, roi_thermal_face_2)[0])
                    test = (255, 0, 255)
                    print('face other')
        except:
            test = (70, 225, 25)
            print('fail')
            continue

        cv2.rectangle(frame_thermal,
                      (x - color_width_crop, y - color_height_crop),
                      (x + w - color_width_crop, y + h - color_height_crop),
                      test, 3)

        del roi_thermal_face_result, roi_bb_rgb_result

    frame_thermal = imutils.resize(frame_thermal, width=640)
    frame_color = imutils.resize(frame_color, width=640)
    cv2.imshow('live_video', frame_color)
    cv2.imshow('thermal_video', frame_thermal)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
color_video.stop()
thermal_video.stop()
