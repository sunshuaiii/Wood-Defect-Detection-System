import cv2
import numpy as np
import imutils


# return true if has dead knot
def dead_knot(frame):
    print("\nDetecting dead knot...")
    frame = imutils.resize(frame, width=1024)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([89, 255, 100])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    knot_number = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 3000 or cv2.contourArea(contour) < 50000:
            continue
        knot_number = knot_number + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Status: {}".format('Dead Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    res1 = cv2.bitwise_and(frame, frame, mask=mask)
    res2 = cv2.bitwise_not(res1)

    # Display Results
    cv2.imshow('Original - Dead Knot', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res2)
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
    # frame = frame + 20
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([89, 255, 100])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    knot_number = 0

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 800 or cv2.contourArea(contour) > 3000:
            continue
        knot_number = knot_number + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Status: {}".format('Small Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    res1 = cv2.bitwise_and(frame, frame, mask=mask)
    res2 = cv2.bitwise_not(res1)

    # Display Results
    cv2.imshow('Original - Small Knot', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res2)
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

    # # crop the image
    cropped = img[:, 120:img.shape[1] - 120]

    # Convert into gray scale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    if np.average(gray) > 140:
        gray = gray - 60
    elif np.average(gray) > 130:
        gray = gray - 30
    elif np.average(gray) > 120:
        gray = gray - 40
    elif np.average(gray) > 100:
        gray = gray - 25
    elif np.average(gray) > 90:
        gray = gray - 20
    if np.average(gray) > 85 or np.average(gray) < 88:
        gray = gray - 15

    # Image processing ( smoothing )
    # Averaging
    blur = cv2.blur(gray, (6, 6))

    # Apply logarithmic transform
    np.seterr(divide='ignore')
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

    # Assume area detected < 14000 is not cracked

    if cv2.countNonZero(gray) < 14000:
        print("No crack")
        return False
    else:
        print("Has crack")
        return True


# return number of holes in the wood
def pinhole(frame):
    print("\nDetecting pinhole...")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    gray = 255 - gray

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

    cv2.imshow('pinhole', imutils.resize(frame, width=1024))
    cv2.waitKey()

    cv2.destroyAllWindows()
    print("Number of holes:", counter)
    return counter


def wood_defect_detection_system():
    image_path = 'imageInput/crack/2.bmp'
    frame = cv2.imread(image_path)
    if frame is None:
        print('Could not open or find the image: ', image_path)
        exit(0)
    print("Reading image from " + image_path)

    # image processing for wood defect detection system

    # 1. size detection
    minimum_width = 3200

    # width < 3200: undersized

    if frame.shape[1] < minimum_width:
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
        # holes = 0: no holes - Grade A

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
