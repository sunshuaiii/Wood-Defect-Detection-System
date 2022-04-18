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

# 1. image preprocessing


# 2. size detection


# 3. dead knot detection / small knots detection
mymodule = SourceFileLoader('knotDetection', 'C:/Users/Alvin/.spyder-py3/Wood-Defect-Detection-System/knot/knotDetection.py').load_module()

#im = cv2.imread('Untitled1.bmp')
frame = cv2.imread('knot1.bmp')
frame2 = cv2.imread('knot2.bmp')
frame3 = cv2.imread('knot3.bmp')
mymodule.knotDetection(frame)
mymodule.knotDetection(frame2)
mymodule.knotDetection(frame3)


# 4. crack detection


# 5. holes / pin detection


# defect detection logic

if size > 100:  # set the minimum size
    if has_dead_knot or has_cracks or holes > 100:   # set the minimum number of holes
        grade = "Grade C"
    elif has_small_knots:
        grade = "Grade B"
    else:
        grade = "Grade A"
else:
    print("Need to resize the wood")
