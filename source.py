from __future__ import print_function
from configparser import Interpolation
import numpy as np
import cv2 as cv
import argparse

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized

def verify_alpha_channel(frame):
    try:
        frame.shape[3] # looking for the alpha channel
    except IndexError:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    return frame

def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0 
    blended = cv.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
    return blended

def apply_circle_focus_blur(frame, intensity=0.2):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    y = int(frame_h/2)
    x = int(frame_w/2)

    mask = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    cv.circle(mask, (x, y), int(y/2), (255,255,255), -1, cv.LINE_AA)
    mask = cv.GaussianBlur(mask, (21,21),11 )

    blurred = cv.GaussianBlur(frame, (21,21), 11)
    blended = alpha_blend(frame, blurred, 255-mask)
    frame = cv.cvtColor(blended, cv.COLOR_BGRA2BGR)
    return frame

parser = argparse.ArgumentParser()
parser.add_argument('--face_cascade', help='Path to face cascade.', default='../opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='../opencv/data/third_party/frontalEyes35x16.xml')
parser.add_argument('--nose_cascade', help='Path to nose cascade.', default='../opencv/data/third_party/Nose18x15.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
nose_cascade_name = args.nose_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
nose_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
if not nose_cascade.load(cv.samples.findFile(nose_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#read image
# hat = cv.imread('hat.png')
glasses = cv.imread('glasses.png', -1)
mustache = cv.imread('mustache.png', -1)

camera_device = args.camera
# Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors = 5)

    blur_mask = apply_circle_focus_blur(frame.copy())    

    # change to BGRA
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

    # iterate faces
    for (x, y, w, h) in faces:
        # ROI = region of interest
        roi_gray = gray[y:y+h, x:x+h]  # rectangle frame of the face gray
        roi_color = frame[y:y+h, x:x+h]  # rectangle frame of the face color

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors = 5)  # extract eyes from region of interest
        #iterate eyes
        for(ex, ey, ew, eh) in eyes:
            # cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey:ey+eh, ex:ex+ew]
            # autoscaling, if necessary
            glasses_copy = image_resize(glasses.copy(), width=ew)
            gw, gh, gc = glasses_copy.shape
            # replace pixels from glasses with roi pixels  --  the easy and unefficient way
            for i in range(0, gw):
                for j in range(0, gh):
                    # print(glasses[i, j])  # RGBA value
                    if glasses_copy[i, j][3] != 0:  # alpha 0 -- transparent
                        roi_color[ey + i, ex + j] = glasses_copy[i, j]
        
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache_copy = image_resize(mustache.copy(), width=nw)

            mw, mh, mc = mustache_copy.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    #print(glasses[i, j]) #RGBA
                    if mustache_copy[i, j][3] != 0: # alpha 0
                        roi_color[ny + int(nh/2.0) + i, nx + j] = mustache_copy[i, j]

    # change back to BGR
    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    # show filter on each frame
    cv.imshow('Face detection', frame)

    #if user pressed 'q' break
    if cv.waitKey(1) == ord('q'): # 
        break;

# turn off camera
cap.release()
# close all windows
cv.destroyAllWindows()
