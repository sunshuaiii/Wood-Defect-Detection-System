import cv2
import numpy as np
import imutils


#im = cv2.imread('Untitled1.bmp')
#frame = cv2.imread('Image_20220128121045551.bmp')
def knotDetection(frame):
    frame = imutils.resize(frame, width=1024)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([99, 255, 100])

    mask = cv2.inRange(hsv, lower_red, upper_red)
# 1.
# mask = cv2.bitwise_not(mask)
# 1.
################################################
    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 800:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Status: {}".format('Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# cv2.drawContours(frame, contours, -1, (255,0 , 0), 2)
# cv2.imshow("Video", frame)
################################################

    res1 = cv2.bitwise_and(frame, frame, mask=mask)
    res2 = cv2.bitwise_not(res1)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
# Hori = np.concatenate((frame,mask_rgb,res2 ), axis=1)

# Verti1 = np.concatenate((frame,mask_rgb,res2 ), axis=0)
# Verti2 = np.concatenate((mask_rgb, res1), axis=0)
# Hori = np.concatenate((Verti1, Verti2), axis=1)

# 2.2
#    out.write(Hori)
# 2.2

# Display Results
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res2)
# cv2.imshow('VERTICAL', Verti)
# cv2.imshow('Hori', Hori)
# cv2.imshow('Hori', Hori)
    cv2.waitKey()

    cv2.destroyAllWindows()
