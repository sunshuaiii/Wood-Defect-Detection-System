import cv2
import numpy
from importlib.machinery import SourceFileLoader

image = ""

# to store the data of the wood
size = 0
has_dead_knot = False
has_cracks = False
holes = 0
has_small_knots = False

grade = ""


# image processing for wood defect detection system

# 1. image preprocessing???
# transform to grayscale
# transformation techniques

# 2. size detection???


# 3. dead knot detection / small knots detection
mymodule = SourceFileLoader('knotDetection', 'C:/Users/Alvin/.spyder-py3/Wood-Defect-Detection-System/knot/knotDetection.py').load_module()

#im = cv2.imread('Untitled1.bmp')
frame = cv2.imread('imageInput/knot1.bmp')
frame2 = cv2.imread('imageInput/knot2.bmp')
frame3 = cv2.imread('imageInput/knot3.bmp')
mymodule.knotDetection(frame)
mymodule.knotDetection(frame2)
mymodule.knotDetection(frame3)


# 4. crack detection


# 5. holes / pin detection
img = cv2.imread('pinhole.bmp')
# im = cv2.imread('Image_20220128121045551.bmp')

# resize the image first
def rescaleFrame(frame, scale = 0.35):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

image_resized = rescaleFrame(img)

def pinhole(image_resized):
    gray=cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
    gray=cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('gray',255-gray)

    contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    counter = 0

    for cnt in contours:
        counter += 1
        area = cv2.contourArea(cnt)
    (x, y, w, h) = cv2.boundingRect(cnt)
    if area < 300:
        cv2.drawContours(image_resized,[cnt],0,(255,0,0),2)
        cv2.rectangle(image_resized, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)
        cv2.putText(image_resized, str(counter), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
    

cv2.imshow('im',image_resized)
cv2.waitKey()

# defect detection logic

if size > 100:  # set the minimum size
    if has_dead_knot or has_cracks or holes > 3:  # set the minimum number of holes
        grade = "Grade C"
    elif has_small_knots or holes <= 3:
        grade = "Grade B"
    else:
        grade = "Grade A"
    print(grade)
else:
    print("Need to resize the wood")
