import cv2

# Opens the Video file
color_url = 'D:\\thermal\\MD_2020-11-05\\sklep\\20000101_180402_channelV.avi'
thermal_url = 'D:\\thermal\\MD_2020-11-05\\sklep\\20000101_180402_channelT.avi'

input_list = list([color_url, thermal_url])
file_names = list(['color', 'thermal'])
for y, z in zip(input_list, file_names):
    i = 0
    cap = cv2.VideoCapture(y)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False or i > 0:
            break
        cv2.imwrite('D:\\thermal\\MD_2020-11-05\\sklep\\' + str(z) + '.jpg', frame)
        i = i + 1
        cap.release()
        cv2.destroyAllWindows()