import cv2
import numpy
from importlib.machinery import SourceFileLoader
mymodule = SourceFileLoader('knotDetection', 'C:/Users/Alvin/.spyder-py3/Wood-Defect-Detection-System/knot/knotDetection.py').load_module()

#im = cv2.imread('Untitled1.bmp')
frame = cv2.imread('knot1.bmp')
frame2 = cv2.imread('knot2.bmp')
frame3 = cv2.imread('knot3.bmp')
mymodule.knotDetection(frame)
mymodule.knotDetection(frame2)
mymodule.knotDetection(frame3)