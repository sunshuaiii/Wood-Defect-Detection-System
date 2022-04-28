import cv2
import numpy as np
import imutils


# undersized: to determine the wood color, brown and orange are not acceptable
# return true if undersized
def undersized(frame):
    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for brown and orange
    lower_bound = np.array([20, 30, 20])
    upper_bound = np.array([90, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    area_detected = cv2.countNonZero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    percentage = area_detected / total_area
    print("Area detected: ", area_detected)
    print("Total area: ", total_area)
    print("Percentage: ", percentage)

    # Assume area detected < 120000 is not undersized

    if percentage < 0.12:
        return False
    else:
        return True


def wood_defect_detection_system():
    print("undersized")
    image_path = 'imageInput/undersized/1.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/undersized/2.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/undersized/3.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/undersized/4.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    print("pinhole")
    image_path = 'imageInput/pinhole/1.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/pinhole/2.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/pinhole/3.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/pinhole/4.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/pinhole/5.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/pinhole/6.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    print("crack")
    image_path = 'imageInput/crack/1.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    # image_path = 'imageInput/crack/2.bmp'
    # frame = cv2.imread(image_path)
    #
    # if undersized(frame):
    #     print("The wood is undersized.")
    # print()
    #
    # image_path = 'imageInput/crack/3.bmp'
    # frame = cv2.imread(image_path)
    #
    # if undersized(frame):
    #     print("The wood is undersized.")
    # print()

    image_path = 'imageInput/crack/4.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/crack/5.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/crack/6.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/crack/7.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    print("knot")

    image_path = 'imageInput/knot/1.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/knot/2.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()

    image_path = 'imageInput/knot/3.bmp'
    frame = cv2.imread(image_path)

    if undersized(frame):
        print("The wood is undersized.")
    print()


if __name__ == "__main__":
    wood_defect_detection_system()
