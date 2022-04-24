# importing necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils

# read a cracked sample image
img = cv2.imread('imageInput/crack/1.bmp')

# new_image = cv2.convertScaleAbs(img, alpha=1, beta=20)

# Convert into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(np.average(gray))
if np.average(gray) > 115:
    gray = gray - 30

# Image processing ( smoothing )
# Averaging
blur = cv2.blur(gray, (1, 1))

# Apply logarithmic transform
img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255

# Specify the data type
img_log = np.array(img_log, dtype=np.uint8)

# Image smoothing: bilateral filter
bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

# Canny Edge Detection
edges = cv2.Canny(bilateral, 100, 200)

# Morphological Closing Operator
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Create feature detecting method
# sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1000)

# Make featured Image
keypoints, descriptors = orb.detectAndCompute(closing, None)
featured_img = cv2.drawKeypoints(closing, keypoints, None)

cv2.imshow('Original', imutils.resize(img, width=1024))
cv2.imshow("img_log", imutils.resize(img_log, width=1024))
cv2.imshow('Output', imutils.resize(featured_img, width=1024))
cv2.waitKey()

# # Create an output image
# cv2.imwrite('imageOutput/CrackDetected-7.jpg', featuredImg)
# # Use plot to show original and output image
# plt.subplot(211), plt.imshow(img)
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(212), plt.imshow(featuredImg, cmap='gray')
# plt.title('Output Image'), plt.xticks([]), plt.yticks([])
# # plt.show()
gray = cv2.cvtColor(featured_img, cv2.COLOR_BGR2GRAY)

print(cv2.countNonZero(gray))

if cv2.countNonZero(gray) <= 1500:
    print("No crack")
else:
    print("Has crack")
