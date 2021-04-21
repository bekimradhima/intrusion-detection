import numpy as np
import cv2
import time

def L2(img1, img2):
    sq_dist = (img1 - img2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff

def background_initialization(bg, n, cap, count):
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            bg.append(frame)
            count += 1
            # print(count)
        else:
            break
    cap.release()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [bg_inter, count]

def skip_background(contours, frame, final, shift1, shift2, index, thresh):
    # ignore contours that are part of the background

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    print(index)
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(0, 0))
    shift1=cv2.erode(shift1,kernel,iterations=3)
    cv2.imshow('c',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    shift2=cv2.dilate(shift2, kernel, iterations=10)
    final2 = cv2.dilate(final, kernel, iterations=3)
    shift2=shift2-final2
    cv2.imshow('d', shift2)
    external_median = np.median(frame[shift1 > 0], overwrite_input=True)
    print(external_median)
    internal_median = np.median(frame[shift2 > 0], overwrite_input=True)
    print(internal_median)

    if np.abs(external_median - internal_median) < thresh:
        return True

def object_detector(contours, index, mask):
    # detect, classify and log objects moving in the frame
    if (500 < cv2.contourArea(contours[index]) < 5000):
        cv2.drawContours(mask, contours, index,[0, 255, 0], -1)

interpolation = np.median
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 30
distance = L2
N_frames = 50
count=0
bg=[]

cap = cv2.VideoCapture('1.avi')
[bg, count] = background_initialization(bg, N_frames, cap, count)

def change_detection(video_path, bg, threshold):
    ftime=True
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        mask = distance(gray, bg) > threshold
        mask = mask.astype(np.uint8) * 255
        cv2.imshow('mask', mask)

        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
        final = cv2.dilate(open, kernel, iterations=3)
        cv2.imshow('morph', final)


        #Background update
        hist, bins = np.histogram(final.flatten(), 256, [0, 256])
        # update background when ligth changes
        if ftime == False:
            if hist[255] > 1.1 * prevhist:
                cv2.accumulateWeighted(gray, bg, 0.05)
                print('change_updated ')


            elif hist[255] < 0.1 * prevhist:
                cv2.accumulateWeighted(gray, bg, 0.3)
                print('selective updated')

        prevhist = hist[255]
        ftime = False
        #find contours
        _, contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        image_external = np.zeros(final.shape, np.uint8)
        colored_contours = np.zeros(frame.shape)
        original_contour = np.zeros(final.shape, np.uint8)
        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
             #person detector
             if cv2.contourArea(cnt)>6000:
                 #draw person in blue
                cv2.drawContours(frame, contours,i,[255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            if cv2.contourArea(contours[j]) < 100 or cv2.contourArea(contours[j]) > 2000:
                continue
            elif skip_background(contours, frame, final , shift1, shift2, j, 40) == True:
                #draw false object in red
                cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
            else :
                #draw true objects in green
                cv2.drawContours(frame, contours, j,[0, 255, 0], -1)


        cv2.imshow('contours', frame)
        time.sleep(0.02)

        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr)
