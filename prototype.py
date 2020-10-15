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
        mask = cv2.inRange(roi, lower, upper)
        output = cv2.bitwise_and(roi, roi, mask=mask)

        (red_thermal_output, green_thermal_output, blue_thermal_output) = cv2.split(output)
        return np.max(red_thermal_output)


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

    # face detection for-loop & selecting faces with temp > bb
    for (x, y, w, h) in faces:
        roi_thermal_face = rgb_thermal[y - color_height_crop: y + h - color_height_crop + 250,
                           x - color_width_crop: x + w - color_width_crop]

        # for (lower, upper) in boundary:
        #     lower = np.array(lower, dtype="uint8")
        #     upper = np.array(upper, dtype="uint8")
        #     mask = cv2.inRange(roi_thermal_face, lower, upper)
        #     output = cv2.bitwise_and(roi_thermal_face, roi_thermal_face, mask=mask)
        #     output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        #     cv2.imshow("images", np.hstack([roi_thermal_face, output]))
        #     (red_thermal_output, green_thermal_output, blue_thermal_output) = cv2.split(output)
        #     print(np.max(red_thermal_output), np.max(green_thermal_output), np.max(blue_thermal_output))

        try:
            if masking(roi_thermal_face, boundary) < masking(roi_bb_rgb, boundary):
                test = (0, 255, 0)
            else:  # face hotter than bb
                test = (0, 0, 255)
        except:
            test = (255, 255, 255)
            continue

        cv2.rectangle(frame_thermal,
                      (x - color_width_crop, y - color_height_crop),
                      (x + w - color_width_crop, y + h - color_height_crop),
                      test, 3)

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
