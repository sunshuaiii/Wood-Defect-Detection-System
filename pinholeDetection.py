import cv2
import numpy as np

im = cv2.imread('imageInput/Untitled1.bmp')
# im = cv2.imread('Image_20220128121045551.bmp')

def pinholeDetection (im) :
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('gray', 255 - gray)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    counter = 0

    for cnt in contours:
        counter += 1
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
    if area < 300:
        cv2.drawContours(im, [cnt], 0, (255, 0, 0), 2)
        cv2.rectangle(im, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)
        cv2.putText(im, str(counter), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('im', im)
    cv2.waitKey()
