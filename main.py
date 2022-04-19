import cv2
import numpy as np
import imutils

frame = cv2.imread('imageInput/knot4.bmp')

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
def deadknot(frame):
    frame = imutils.resize(frame, width=1024)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([99, 255, 100])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 1:
        return False;
    else:
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 800:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Status: {}".format('Dead Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            res1 = cv2.bitwise_and(frame, frame, mask=mask)
            res2 = cv2.bitwise_not(res1)

    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # Hori = np.concatenate((frame,mask_rgb,res2 ), axis=1)

    # Verti1 = np.concatenate((frame,mask_rgb,res2 ), axis=0)
    # Verti2 = np.concatenate((mask_rgb, res1), axis=0)
    # Hori = np.concatenate((Verti1, Verti2), axis=1)

    # Display Results
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res2)
    # cv2.imshow('VERTICAL', Verti1)
    # cv2.imshow('Hori', Hori)
    # cv2.imshow('Hori', Hori)
            cv2.waitKey()

            cv2.destroyAllWindows()
            return True
    
def smallknot(frame):
    # resize img and chg to RGB
    frame = imutils.resize(frame, width=1024)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lower_red = np.array([0, 0, 0])
    upper_red = np.array([70, 255, 100])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    _, thresh = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
     
    if len(contours) < 1:
        return False;
    else:
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 800:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Status: {}".format('Small Knot'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            res1 = cv2.bitwise_and(frame, frame, mask=mask)
            res2 = cv2.bitwise_not(res1)

    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # Hori = np.concatenate((frame,mask_rgb,res2 ), axis=1)

    # Verti1 = np.concatenate((frame,mask_rgb,res2 ), axis=0)
    # Verti2 = np.concatenate((mask_rgb, res1), axis=0)
    # Hori = np.concatenate((Verti1, Verti2), axis=1)

    # Display Results
            cv2.imshow('frame', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('res', res2)
    # cv2.imshow('VERTICAL', Verti1)
    # cv2.imshow('Hori', Hori)
    # cv2.imshow('Hori', Hori)
            cv2.waitKey()

            cv2.destroyAllWindows()
            return True
    # if contours

# 4. crack detection


# 5. holes / pin detection
img = cv2.imread('imageInput/pinhole.bmp')
# im = cv2.imread('Image_20220128121045551.bmp')

# resize the image first
def rescaleFrame(frame, scale = 0.35):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# image_resized = rescaleFrame(img)

def pinhole(image_resized):
    img1 = image_resized.copy()
    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray=cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]
    cv2.imshow('gray',255-gray)
    
    holes,hierarchy = cv2.findContours(gray,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE) 
    print("number of holes:",len(holes))
    counter = 0
    
    for cnt in holes:
        counter += 1
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area < 300:
            cv2.drawContours(image_resized,[cnt],0,(255,0,0),2)
            cv2.rectangle(img1, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 2)
            cv2.putText(img1, str(counter), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
            
        cv2.imshow('im',image_resized)
        cv2.waitKey()
        cv2.destroyAllWindows()    
        return holes
    


# defect detection logic
rescaled_img = rescaleFrame(frame)
holes = pinhole(rescaled_img)
has_dead_knot = deadknot(rescaled_img)
has_small_knots = smallknot(rescaled_img)
if size < 100:  # set the minimum size
    if has_dead_knot or has_cracks or holes > 3:  # set the minimum number of holes
        grade = "Grade C"
    elif has_small_knots or holes <= 3:
        grade = "Grade B"
    else:
        grade = "Grade A"
    print(grade)
else:
    print("Need to resize the wood")
print(grade)
