import numpy as np
import cv2
import time


def L1(img1, img2):
    diff = np.abs(img1 - img2)
    print(len(img1.shape))
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff


def L2_norm(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    newimg1 = (img1 - mean1) / np.std(img1)
    newimg2 = (img2 - mean2) / np.std(img2)
    sq_dist = (newimg1 - newimg2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff


def Linf(F1, F2):
    diff = np.abs(F1 - F2)
    if F1.shape[-1] == 3 and len(F1.shape) == 3:
        diff = np.max(diff, axis=-1)
    return diff


def linear_stretching(img, max_value, min_value):
    img[img < min_value] = min_value
    img[img > max_value] = max_value
    linear_stretched_img = 255. / (max_value - min_value) * (img - min_value)
    return linear_stretched_img


def exponential_operator(img, r):
    exp_img = ((img / 255) ** r) * 255
    return exp_img


def twoframedifference(frame, previousframe, distance_type, threshold):
    distance = distance_type(frame, previousframe)
    maskbool = distance > threshold
    mask = maskbool.astype(np.uint8) * 255
    return [mask]


def threeframedifference(frame, prev1, prev2, distance_type, threshold):
    if distance_type == "L1":
        [maskbool, mask1] = twoframedifference(frame, prev1, distance_type, threshold)
        if len(prev2) != 0:
            [maskbool1, mask2] = twoframedifference(prev1, prev2, distance_type, threshold)
    mask = np.logical_and(maskbool, maskbool1)
    prev1[np.logical_not(mask)] = np.array([255, 255, 255])
    mask = mask.astype(np.uint8) * 255


# Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean


def selective_inizializaion(cap, n):
    idx = 0
    selective_bg=[]
    while cap.isOpened() and idx < n:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret or not frame is None:
            idx += 1
            if idx == 1:
                pgray = np.copy(gray)
                # bg=sgray.astype(np.uint8)
            else:
                smask = (distance(gray, pgray) > 5)
                smask = smask.astype(np.uint8) * 255
                cv2.imshow('premask', smask)
                pgray = gray
                sopen = cv2.morphologyEx(smask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                         iterations=2)
                sdilate = cv2.dilate(sopen, None, iterations=3)
                sinv = 255 - sdilate
                sbg = np.copy(gray)
                sbg[np.logical_not(sinv)] = np.asarray(0)
                #if idx % 2 == 0 :
                selective_bg.append(sbg)
                #sbg1 = np.copy(sbg)
                #cv2.imshow('prem', sbg1)
        else:
            print('done1')
            break
    cap.release()
    print('done')
    selective = np.stack(selective_bg, axis=0)
    selective = interpolation(selective, axis=0)
    selective = selective.astype(np.uint8)
    bg = selective

    return bg



def background_initialization(bg, n, cap, count):
    n = n * 2
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            # Release the Video if ret is false
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hist,bins = np.histogram(frame.flatten(),256,[0,256])
            # eq_op = pfm(hist)*255
            # frame = eq_op[frame]
            # frame
            # if count == 0:
            #    bg.append(frame)
            # else:
            #    bg.append(alfa * frame)
            # bg[count] = (1 - alfa) * bg[count - 1] + bg[count]
            if (count % 2 != 0):
                # frame = cv2.GaussianBlur(frame, (5, 5), 0)
                bg.append(frame)
            count += 1
            # print(count)
        else:
            break
    cap.release()
    b = bg.copy()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    # bg_inter = linear_stretching(np.copy(bg_inter), 255,250)
    # bg_inter = cv2.GaussianBlur(bg_inter, (5, 5), 0)
    cv2.destroyAllWindows()
    return [b, bg_inter, count]


def selective_background_update(bg1, frame, prev_bg, alfa, closing):
    frame[np.logical_not(closing)] = np.asarray(0)
    # cv2.imshow('g1', frame)

    bg2 = np.copy(prev_bg)
    bg3 = np.copy(prev_bg)
    prev_bg = prev_bg.astype(np.uint8)
    prev_bg[np.logical_not(closing)] = np.asarray(0)
    # cv2.imshow('g2', prev_bg)
    bg3[closing == 255] = np.asarray(0)
    bg2 = (1 - alfa) * prev_bg + alfa * frame + bg3
    bg1 = np.copy(bg2)
    # cv2.imshow('g4', bg2.astype(np.uint8))
    return bg1


def skip_background(contours, frame, final, shift1, shift2, index, thresh):
    # ignore contours that are part of the background
    # take two shifted contours, add them and mask using original contours to obtain internal contour
    # print(index)
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(0, 0))
    shift1 = cv2.erode(shift1, kernel, iterations=5)
    # cv2.imshow('internal',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    # shift2=cv2.dilate(shift2, kernel, iterations=5)
    shift2 = shift2 - final
    # shift2=cv2.erode(shift2,kernel,iterations=4)
    # cv2.imshow('external', shift2)
    external_median = (frame[shift1 > 0])
    hist = cv2.calcHist([external_median], [0], None, [256], [0, 256])
    internal_median = (frame[shift2 > 0])
    hist1 = cv2.calcHist([internal_median], [0], None, [256], [0, 256])
    # print('internal %d',internal_median)
    compare = cv2.compareHist(hist, hist1, cv2.HISTCMP_CORREL)
    # print(compare)
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
N_frames = 40  # then refresh
# blob detector parameters

cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
bg = selective_inizializaion(cap, N_frames)

# fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)
file = open("detected_log.txt", "w+")


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def change_detection(video_path, bg, threshold, frame, b):
    # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    frame_number = 0
    # bg_inter1=[]
    # prevhist2 = 0
    cond = False
    # bg7=bg.astype(np.uint8)
    # cond2 = False
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
        cv2.imshow('bg', bg)

        # Compute background suptraction
        mask = (distance(gray, bg) > 0.5)
        # m = distance(gray, bg)
        # mask = (m > np.mean(m))
        # m  = NCC(gray,bg.astype(np.uint8))
        # mask7 = m
        # mask7 = (mask7.astype(np.uint8) * 255)
        mask = mask.astype(np.uint8) * 255
        # mask= fgbg.apply(gray)
        cv2.imshow('mask', mask)
        # cv2.imshow('mask7', mask7.astype(np.uint8))
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        # cv2.imshow('Blur', blur)
        # blur2 = cv2.filter2D(blur,-1,denoising_kernel)
        # blur2=cv2.fastNlMeansDenoising(blur)
        # cv2.imshow('Blur2', blur2)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        # im_out = thresh | im_floodfill_inv
        # cv2.imshow('combine', im_out)
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                   iterations=1)
        cv2.imshow('opening', opening)
        dilated = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=3)
        # dilated2 = cv2.bitwise_not(dilated)
        # clos = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        cv2.imshow("clos", dilated)
        closing = dilated
        inv_closing = 255 - closing
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 5)
        cv2.imshow('inv_closing', inv_closing)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        # edges = gray.astype(np.uint8)
        mask[np.logical_not(closing)] = np.asarray(0)
        blur[np.logical_not(closing)] = np.asarray(0)
        mask6 = (distance(gray, bg) > 0.25)
        # mask6 = m > np.mean(m)/2
        mask6 = mask6.astype(np.uint8) * 255
        mask6[np.logical_not(closing)] = np.asarray(0)
        blur6 = cv2.GaussianBlur(mask6, (5, 5), 3)
        ret6, thresh6 = cv2.threshold(blur6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('mask6', mask6)
        # cv2.imshow('blur6', blur6)
        cv2.imshow('thresh6', thresh6)

        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                    iterations=1)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                    iterations=2)
        #
        # cv2.imshow('e1', edges2)
        cv2.imshow('e2', opening2)
        cv2.imshow('e22', closing2)
        f = gray.astype(np.uint8)
        f[np.logical_not(closing2)] = np.asarray(0)
        out = closing2
        im2 = gray.copy()
        closing4 = cv2.dilate(closing2, None, iterations=2)
        cv2.imshow('closing4', closing4)
        closing3 = 255 - closing4
        im2[np.logical_not(closing3)] = np.asarray(0)
        cv2.imshow('im2', im2)

        hist, bins = np.histogram(thresh6.flatten(), 256, [0, 256])
        # update background when ligth changes

        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        final = out
        blob_count = len(contours)

        if (hist[255] < 0.2 * prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.3, closing3)
            # bg = background_update(bg1, im2, bg, 0.1)
            print('background update')
        # cv2.resizeWindow('contours', 500, 500)
        # image_external = np.zeros(final.shape, np.uint8)
        # colored_contours = np.zeros(frame.shape)
        # original_contour = np.zeros(final.shape, np.uint8)
        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
            # person detector
            # epsilon = 0.001*cv2.arcLength(contours[i],True)
            # contours[i] = cv2.approxPolyDP(contours[i],epsilon,True)
            if cv2.contourArea(cnt) > 4500:
                # draw person in blue
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                file.write("frame %d, detected person, blob area: %d, blob perimeter: %d\r\n" % (
                frame_number, area, perimeter))
                cv2.drawContours(frame, contours, i, [255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            # if  cv2.contourArea(contours[j]) < 100 or cv2.contourArea(contours[j]) > 2000:
            #    continue
            if (450 < cv2.contourArea(contours[j]) < 1500):
                if skip_background(contours, frame, final, shift1, shift2, j, 0.9) == True:
                    # draw false object in red
                    area1 = cv2.contourArea(cnt)
                    perimeter1 = cv2.arcLength(cnt, True)
                    file.write("frame %d, detected FALSE book, blob area: %d, blob perimeter: %d\r\n" % (
                    frame_number, area1, perimeter1))
                    cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
                else:
                    # draw true objects in green
                    area2 = cv2.contourArea(cnt)
                    perimeter2 = cv2.arcLength(cnt, True)
                    file.write("frame %d, detected REAL book, blob area: %d, blob perimeter: %d\r\n" % (
                    frame_number, area2, perimeter2))
                    cv2.drawContours(frame, contours, j, [0, 255, 0], -1)

        cv2.imshow('contours', frame)
        time.sleep(0.02)

        if (cond == True and hist[255] > 1.0989 * prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
            print('change_updated')
        elif (cond == True and blob_count == 0):
            bg = selective_background_update(bg1, gray, bg, 0.3, closing3)
            print('background update')
        elif (cond == True and (prevclos == closing).all == True):
            diff = prevclos - closing
            cv2.imshow('diff', diff)
            bg = selective_background_update(bg1, gray, bg, 0.3, diff)

        prevhist = hist[255]
        prevclos = closing
        cond = True
        frame_number += 1
        # prevfr=gray3-im
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, frame, b)