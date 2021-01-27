import cv2
import imutils
from threading import Thread
import numpy as np

#print(cv2.getBuildInformation())
#C:\\Users\\Dave\\Videos\\Any Video Converter\\MOV\\20000101_180402_channelT_x264.mov
color_url = 'D:\\thermal\\MD_2020-11-05\\sklep\\20000101_180402_channelV.avi'
thermal_url = 'D:\\thermal\\MD_2020-11-05\\sklep\\20000101_180402_channelT.avi'

face_cascade = cv2.CascadeClassifier('data\\haarcascades\\haarcascade_frontalface_default.xml')
color_video = cv2.VideoCapture(color_url)
thermal_video = cv2.VideoCapture(thermal_url)

frame_count = 0
face_frame_count = 0

#color_video = WebcamVideoStream(src=color_url).start()
#thermal_video = WebcamVideoStream(src=thermal_url).start()

color_width_original = 1920
color_height_origial = 1080
color_width_crop = 320
color_height_crop = 60

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

boundary = [([230, 0, 0], [255, 40, 40])] #rgb
roi_bb_capture = None
roi_thermal_capture = None

while True:
    a, frame_color = color_video.read()
    b, frame_thermal = thermal_video.read()
    rgb_thermal = cv2.cvtColor(frame_thermal, cv2.COLOR_BGR2RGB)

    # faces = face_cascade.detectMultiScale(frame_color,
    #                                       scaleFactor=1.3,
    #                                       minNeighbors=5)

    # crop color frame to size roughly of thermal frame
    frame_color = frame_color[color_height_crop:color_height_origial - color_height_crop,
                  color_width_crop:color_width_original - color_width_crop]

    #rescale frames


    # select area of black-body on thermal camera
    while roi_bb_capture == None:
        roi_bb_capture = cv2.selectROI(frame_thermal)
        #roi_bb_rgb = rgb_thermal[int(roi_bb[1]):int(roi_bb[1] + roi_bb[3]), int(roi_bb[0]):int(roi_bb[0] + roi_bb[2])]
        cv2.destroyAllWindows()

    while roi_thermal_capture == None:
        roi_thermal_capture = cv2.selectROI(frame_thermal)
        x, y, w, h = roi_thermal_capture
        #roi_check_rgb = rgb_thermal[int(roi_check[1]):int(roi_check[1] + roi_check[3]), int(roi_check[0]):int(roi_check[0] + roi_check[2])]
        #print(roi_check)
        cv2.destroyAllWindows()

    throat_correction = 0 # used to include chin and throat in roi_thermal_face
    # face detection for-loop & selecting faces with temp > bb
    # for (x, y, w, h) in roi_check:
    # roi_thermal_face = rgb_thermal[y - color_height_crop: y + h - color_height_crop + throat_correction,
    #                    x - color_width_crop: x + w - color_width_crop]

    roi_bb_capture_frame = rgb_thermal[int(roi_bb_capture[1]):int(roi_bb_capture[1] + roi_bb_capture[3]), int(roi_bb_capture[0]):int(roi_bb_capture[0] + roi_bb_capture[2])]
    thermal_roi_capture_frame = rgb_thermal[int(roi_thermal_capture[1]):int(roi_thermal_capture[1] + roi_thermal_capture[3]), int(roi_thermal_capture[0]):int(roi_thermal_capture[0] + roi_thermal_capture[2])]

    roi_bb_rgb_result = masking(roi_bb_capture_frame, boundary)
    roi_thermal_result = masking(thermal_roi_capture_frame, boundary)

    print(roi_bb_rgb_result)
    print(roi_thermal_result)
    try:
        if ((roi_thermal_result < roi_bb_rgb_result) and (roi_bb_rgb_result > 200)):
            # bb is hottest & has 'red mark'
            test = (0, 255, 0)
            print('bb')
        elif ((roi_thermal_result > roi_bb_rgb_result) and (roi_thermal_result > 200)):
            # face is hottest and has 'red mark'
            test = (0, 0, 255)
            print('face')
        elif ((roi_thermal_result == roi_bb_rgb_result) and (roi_thermal_result == 0)):
            # no red mark in roi's
            roi_thermal_face_2 = (x, y, w, h + throat_correction)
            if minmaxlox(frame_thermal, roi_bb_capture)[0] < minmaxlox(frame_thermal, roi_thermal_face_2)[0]:
                # bb is hotter but has no 'red mark'
                #print(minmaxlox(frame_thermal, roi_bb_capture)[0])
                #print(minmaxlox(frame_thermal, roi_thermal_face_2)[0])
                test = (0, 255, 0)
                print('bb other')
            else:
                # face is hotter but has no 'red mark'
                #print(minmaxlox(frame_thermal, roi_bb_capture)[0])
                #print(minmaxlox(frame_thermal, roi_thermal_face_2)[0])
                test = (255, 0, 255)
                print('face other')
    except:
        test = (70, 225, 25)
        print('fail')
        continue
    print(test)
    cv2.rectangle(frame_color,
                  (int(x), int(y)),
                  (int(x + w), int(y + h)),
                  (50, 50, 50), 5)
    cv2.rectangle(frame_thermal,
                  (int(x)+10, int(y)+10),
                  (int(x + w) + 10, int(y + h) + 10),
                  test, 3)

    del roi_thermal_result, roi_bb_rgb_result, test
    # try:
    #     del roi_thermal_face_2
    # except:
    #     continue

    #frame_thermal = imutils.resize(frame_thermal, width=640)
    #frame_color = imutils.resize(frame_color, width=640)
    cv2.imshow('live_video', frame_color)
    cv2.imshow('thermal_video', frame_thermal)
    if cv2.waitKey(int(1/60*1000)) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
#color_video.stop()
#thermal_video.stop()
