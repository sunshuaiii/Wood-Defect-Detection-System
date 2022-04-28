import cv2
import numpy as np
import imutils


# undersized: to determine the wood color, brown and orange are not acceptable
# return true if undersized
def undersized(frame):
    # minimum_width = 3200
    #
    # longer = 0
    # if frame.shape[1] > frame.shape[0]:
    #     longer = frame.shape[1]
    # else:
    #     longer = frame.shape[0]
    #
    # if longer > minimum_width:
    #     return False
    # else:
    #     return True
    frame = imutils.resize(frame, width=1024)
    # blur = cv2.blur(frame, (4, 4))
    # cv2.imshow("blur", blur)

    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsv)

    # lower bound and upper bound for brown color
    lower_bound = np.array([15, 100, 20])
    upper_bound = np.array([90, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    cv2.imshow("Image", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    area_detected = cv2.countNonZero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    print("Area detected: ", area_detected)
    # print("Total area: ", total_area)
    # print("Percentage: ", area_detected/total_area)

    # Assume area detected < 1100 is not cracked

    if cv2.countNonZero(mask) < 120000:
        return False
    else:
        return True


# return true if has dead knot
def dead_knot(frame):
    print("\nDetecting dead knot...")
    # resize the pic
    frame = imutils.resize(frame, width=1024)
    # chg image from BGR to RGB
    RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # apply Gaussian to the image
    blur = cv2.GaussianBlur(RGB_img, (3, 3), cv2.BORDER_DEFAULT)

    # increase the brightness
    if np.average(blur) < 82:
        blur = blur + 40
    elif np.average(blur) < 100:
        blur = blur + 30

    # define the lower and upper boundary for the red color to be used as mask
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([99, 255, 100])
    # create a mask of the image which has the contrast of color
    mask = cv2.inRange(blur, lower_red, upper_red)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    knot_number = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 2500:
            continue
        knot_number = knot_number + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Status: {}".format('Dead Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display Results
    cv2.imshow('Original - Dead Knot', frame)
    cv2.imshow('Average', blur)
    cv2.imshow('mask', mask)
    cv2.waitKey()

    cv2.destroyAllWindows()

    if knot_number == 0:
        print("No dead knot")
        return False
    else:
        print("Dead knots: ", knot_number)
        return True


# return true if has small knots
def small_knot(frame):
    print("\nDetecting small knots...")
    # resize img and chg to RGB
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), cv2.BORDER_DEFAULT)
    if np.average(blur) > 120:
        blur = blur - 30

    # create a mask of the image where the ROI lies in between the range specified
    mask = cv2.inRange(blur, 70, 88)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    knot_number = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 300 or cv2.contourArea(contour) > 500:
            continue
        knot_number = knot_number + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Status: {}".format('Small Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # res1 = cv2.bitwise_and(frame, frame, mask=mask)
    # res2 = cv2.bitwise_not(res1)

    # Display Results
    cv2.imshow('Original - Small Knot', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('average', blur)
    # cv2.imshow('res', res2)
    cv2.waitKey()

    cv2.destroyAllWindows()

    if knot_number == 0:
        print("No small knot")
        return False
    else:
        print("Small knots: ", knot_number)
        return True


def crack(img):
    print("\nDetecting cracks...")

    # crop the image
    cropped = img[:, 120:img.shape[1] - 120]

    # Convert into gray scale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if np.average(gray) > 140:
        gray = gray - 70
    # elif np.average(gray) > 130:
    #     gray = gray - 30
    elif np.average(gray) > 120:
        gray = gray - 50
    elif np.average(gray) > 110:
        gray = gray - 30
    elif np.average(gray) > 100:
        gray = gray - 40
    elif np.average(gray) > 90:
        gray = gray - 35
    elif np.average(gray) > 80:
        gray = gray - 20

    # Image processing ( smoothing )
    # Averaging
    blur = cv2.blur(gray, (4, 4))

    # Apply logarithmic transform
    np.seterr(divide='ignore')
    img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255

    # Specify the data type
    img_log = np.array(img_log, dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 8, 75, 75)

    # Canny Edge Detection
    edges = cv2.Canny(bilateral, 280, 280)

    # Morphological Closing Operator
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Create feature detecting method
    orb = cv2.ORB_create(nfeatures=1500)

    # Make featured Image
    key_points, descriptors = orb.detectAndCompute(closing, None)
    featured_img = cv2.drawKeypoints(closing, key_points, None)

    cv2.imshow('Original - Crack', imutils.resize(img, width=1024))
    cv2.imshow("img_log", imutils.resize(img_log, width=1024))
    cv2.imshow('Output', imutils.resize(featured_img, width=1024))
    cv2.waitKey()
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(featured_img, cv2.COLOR_BGR2GRAY)
    print("Area detected: ", cv2.countNonZero(gray))

    # Assume area detected < 1100 is not cracked

    if cv2.countNonZero(gray) < 1100:
        print("No crack")
        return False
    else:
        print("Has crack")
        return True


# rescale the frame
def rescale_frame(frame, scale=0.35):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# return number of holes in the wood
def pinhole(frame):
    print("\nDetecting pinhole...")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    gray = 255 - gray
    cv2.imshow("gray", rescale_frame(gray))

    holes, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0

    for cnt in holes:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area < 0.01 or area > 70:
            continue
        counter += 1
        cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 2)
        cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)
        cv2.putText(frame, str(counter), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('pinhole', rescale_frame(frame))
    cv2.waitKey()

    cv2.destroyAllWindows()
    print("Number of pinholes:", counter)
    return counter


def wood_defect_detection_system():
    # change the image_path to detect various defect types of woods

    image_path = 'imageInput/undersized/2.bmp'
    frame = cv2.imread(image_path)
    if frame is None:
        print('Could not open or find the image: ', image_path)
        exit(0)
    print("Reading image from " + image_path)

    # image processing for wood defect detection system

    # 1. size detection
    if undersized(frame):
        print("The wood is undersized.")
        return
    else:
        # 2. dead knot detection / small knots detection
        has_dead_knot = dead_knot(frame)

        if has_dead_knot:
            grade = "C"
            print("\n\nGrade of the wood: " + grade)
            return

        # 3. crack detection
        has_cracks = crack(frame)
        if has_cracks:
            grade = "C"
            print("\n\nGrade of the wood: " + grade)
            return

        # 4. pinhole detection
        holes = pinhole(frame)

        # holes > 10: many holes - Grade C
        # holes 1-10: less holes - Grade B
        # holes = 0: no holes and no small knot - Grade A

        if holes > 10:  # set the minimum number of holes
            grade = "C"
        elif holes > 0:
            grade = "B"
        else:  # holes == 0
            # 5. small knot detection
            has_small_knots = small_knot(frame)
            if has_small_knots:
                grade = "B"
            else:
                grade = "A"
        print("\n\nGrade of the wood: " + grade)


if __name__ == "__main__":
    wood_defect_detection_system()
