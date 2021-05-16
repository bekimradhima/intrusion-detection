"""
Created on Sun May 16 18:26:18 2021
@author: Iacopo Curti , Filippo Guarda , Bekim Radhima
"""

import numpy as np
import cv2
import time


def L2_norm(img1, img2):
    # Computing the mean value for the distance
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    # each image is obtained subtracting its mean value and dividing by its standard deviation in order to normalize the two images
    newimg1 = (img1 - mean1) / np.std(img1)
    newimg2 = (img2 - mean2) / np.std(img2)
    # here it is commputed the square distance of the previously computed images
    sq_dist = (newimg1 - newimg2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff


def two_frame_difference(frame, previous_frame, distance, th):
    dist = distance(frame, previous_frame)
    mask = dist > th
    return mask


def three_frame_difference(frame, previous_frames, distance, th):
    masks = []
    masks.append(two_frame_difference(frame, previous_frames[1], distance, th))
    if len(previous_frames) > 1:
        masks.append(two_frame_difference(previous_frames[1], previous_frames[0], distance, th))
    and_mask = np.prod(masks, axis=0)
    and_mask = and_mask.astype(np.uint8) * 255
    return and_mask


# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean


def selective_background_initialization3(bg, n, cap, count2):
    # here we define the varibales needed
    thresh = 0.085
    selective = []
    previous_frames = []
    n = n * 2
    count = 0
    while cap.isOpened() and count2 < n:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        # we took into consideration only the 'even' frames in the series of 2n frames, in order to increse the robustness of the background
        if (count2 % 2 != 0):
            if count < 2:
                # initialize previous frames properly
                previous_frames.append(frame.astype(float))
            else:
                frame = frame.astype(float)
                # Compute the mask using three frame difference of the current and the two previous frames
                mask = three_frame_difference(frame, previous_frames, distance, thresh)
                mask = mask.astype(np.uint8)
                # compute the binary morphology
                sopen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                         iterations=1)
                sdilate = cv2.morphologyEx(sopen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
                                           iterations=2)
                sinv = 255 - sdilate
                # distinguish the background with respect to the foreground using the bianry morphology previously computed
                sbg = np.copy(frame)
                sbg[np.logical_not(sinv)] = np.asarray(0)
                # append the selective background to a list in order to do the interpolation of the background at the end
                selective.append(sbg)
                # previous frame is poped out in order to store antoher frame in the list
                previous_frames.pop(0)
                previous_frames.append(frame)
            count += 1
        count2 += 1
        # Stop playing when entered 'q' from keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    # it is computed the final background operating an interpolation(median) of the selective list of background
    bg_inter = np.stack(selective, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return bg_inter


# this is the blind background intilaization that can be used instead of the selctive
def background_initialization(bg, n, cap, count):
    n = n * 2
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret and not frame is None:
            # we took into consideration only the 'even' frames in the series of 2n frames, in order to increse the robustness of the background
            if (count % 2 != 0):
                bg.append(frame)
            count += 1
        else:
            break
    cap.release()
    b = bg.copy()
    # it is computed the final background operating an interpolation(median) of the selective list of background
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [b, bg_inter, count]


def selective_background_update(bg1, frame, prev_bg, alfa, closing):
    # we distinguish the background pixel from foreground pixel
    frame[np.logical_not(closing)] = np.asarray(0)
    bg2 = np.copy(prev_bg)
    bg3 = np.copy(prev_bg)
    prev_bg = prev_bg.astype(np.uint8)
    # computing the background mask eliminating from the image the foreground pixels
    prev_bg[np.logical_not(closing)] = np.asarray(0)
    bg3[closing == 255] = np.asarray(0)
    # we sum using the alpha blending procedure the background image currently used, looking at the pixels considered part of the curernt foreground, to  the current background
    bg2 = (1 - alfa) * prev_bg + alfa * frame + bg3
    bg1 = np.copy(bg2)
    return bg1


def skip_background(contours, frame, final, shift1, shift2, index, thresh):
    # distinguish contours that are part of the background from real objects
    # take image mask and erode to have a sharper detected object
    cv2.drawContours(shift1, contours, index, 255, - 1, offset=(0, 0))
    shift1 = cv2.erode(shift1, kernel, iterations=5)
    # take the contours of the object and subtract the first mask to have the outer contours
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    shift2 = shift2 - final
    # use the masks to find the pixel values at the detected contours and calculate the histograms
    cv2.imshow('external', shift2)
    external_median = (frame[shift1 > 0])
    hist = cv2.calcHist([external_median], [0], None, [256], [0, 256])
    internal_median = (frame[shift2 > 0])
    cv2.imshow('inte', shift1)
    hist1 = cv2.calcHist([internal_median], [0], None, [256], [0, 256])
    # compare the two histograms
    compare = cv2.compareHist(hist, hist1, cv2.HISTCMP_CORREL)
    # return the condition of false object if the two histograms are matching, thus is part of the background
    if compare > thresh:
        return True


###Define change detection parameters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 30
distance = L2_norm
bg = []
b = []
bg1 = []
bg2 = []
frame = []
N_frames = 35  # then refresh

cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
bg = selective_background_initialization3(bg, N_frames, cap, count)
# opening of the file for the blob detection
file = open("detected_log.txt", "w+")


def change_detection(video_path, bg, threshold, frame, b):
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    frame_number = 0
    cond = False
    while (cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break
        # Convert to grayscale and blur frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('bg', bg.astype(np.uint8))

        # Compute background suptraction
        mask = (distance(gray, bg) > 0.5)
        mask = mask.astype(np.uint8) * 255
        cv2.imshow('mask', mask)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                   iterations=1)
        closing = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=3)

        # we use this final dilation to find a ROI for our new mask
        mask6 = (distance(gray, bg) > 0.25)
        mask6 = mask6.astype(np.uint8) * 255
        mask6[np.logical_not(closing)] = np.asarray(0)
        blur6 = cv2.GaussianBlur(mask6, (5, 5), 3)
        ret6, thresh6 = cv2.threshold(blur6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('mask6', mask6)

        # compute the new binary morphology
        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                    iterations=1)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                    iterations=2)
        out = closing2

        # find a rougher morphology for the background updating procedure
        im2 = gray.copy()
        closing4 = cv2.dilate(closing2, None, iterations=2)
        closing3 = 255 - closing4
        im2[np.logical_not(closing3)] = np.asarray(0)

        # find the contours
        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        final = out

        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
            # person detector
            if cv2.contourArea(cnt) > 4500:
                # draw person in blue
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                # log a person detection
                file.write("frame %d, detected person, blob area: %d, blob perimeter: %d\r\n" % (
                frame_number, area, perimeter))
                cv2.drawContours(frame, contours, i, [255, 0, 0], -1)

        for j, cnt in enumerate(contours):

            if (450 < cv2.contourArea(contours[j]) < 1350):
                if skip_background(contours, frame, final, shift1, shift2, j, 0.9) == True:
                    # draw false object in red
                    area1 = cv2.contourArea(cnt)
                    perimeter1 = cv2.arcLength(cnt, True)
                    # log a ghost image detection
                    file.write("frame %d, detected FALSE book, blob area: %d, blob perimeter: %d\r\n" % (
                    frame_number, area1, perimeter1))
                    cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
                else:
                    # compute the extent parameter for the contour
                    x, y, w, h = cv2.boundingRect(cnt)
                    rect_area = w * h
                    area2 = cv2.contourArea(cnt)
                    perimeter2 = cv2.arcLength(cnt, True)
                    extent = float(area2) / rect_area
                    if (extent > 0.7):
                        # log a real object detection
                        file.write(
                            "frame %d, detected REAL book, blob area: %d, blob perimeter: %d, blob extent: %f\r\n" % (
                            frame_number, area2, perimeter2, extent))
                        cv2.drawContours(frame, contours, j, [0, 255, 0], -1)

        cv2.imshow('contours', frame)
        hist, bins = np.histogram(thresh6.flatten(), 256, [0, 256])
        # update the histogram where there is less motion with respect to the background
        if (cond == True and hist[255] < 0.2 * prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.3, closing3)
        # update background when ligth changes, so if there is a change in the histogram computed starting form the rougher morphology
        elif (cond == True and hist[255] > 1.0999 * prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
        # the following condition never happen in this video but it is usseful to integrate new objects which are fixed and not moving in the background
        elif (cond == True and (closing == prevclos).all == True):
            diff = prevclos - closing
            cv2.imshow('diff', diff)
            bg = selective_background_update(bg1, gray, bg, 0.7, diff)

        prevhist = hist[255]
        prevclos = closing
        cond = True
        frame_number += 1
        time.sleep(0.02)
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


# We use the change detection function here
change_detection('1.avi', bg, thr, frame, b)
file.close()