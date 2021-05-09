import numpy as np
import cv2
import time


kernelq=np.array([[4, 3, 2, 3, 4],
       [3, 2, 1, 2, 3],
       [2, 1, 0, 1, 2],
       [3, 2, 1, 2, 3],
       [4, 3, 2, 3, 4]], dtype=np.uint8)/100

def L2(img1, img2):
    sq_dist = (img1 - img2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff

def exponential_operator(img, r):
    exp_img = ((img/255)**r) * 256
    return exp_img


def background_initialization(bg, n, cap, count):
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            #frame = cv2.GaussianBlur(frame, (7,7), 1)
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

def skip_background(contours, frame, mask, shift1, shift2, index, thresh):
    # ignore contours that are part of the background

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    #print(index)
    cv2.drawContours(shift1, contours, index, 255, -1, offset=(0, 0))
    shift1=cv2.erode(shift1,kernel,iterations=4)
    #cv2.imshow('internal',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    #shift2=cv2.dilate(shift2, kernel, iterations=5)
    shift2=shift2-mask
    #shift2=cv2.erode(shift2,kernel,iterations=4)
    #cv2.imshow('external', shift2)
    internal_contour =(frame[shift1 > 0])
    hist = cv2.calcHist([internal_contour], [0], None, [256], [0, 256])
    external_contour =(frame[shift2 > 0])
    hist1 = cv2.calcHist([external_contour], [0], None, [256], [0, 256])
    #print('internal %d',internal_median)
    compare= cv2.compareHist(hist, hist1, cv2.HISTCMP_CORREL)
    #print(compare)
    if compare > thresh:
        return True

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


interpolation = np.median
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 35
distance = L2
N_frames = 50
count=0
bg=[]
cap = cv2.VideoCapture('1.avi')
[bg, count] = background_initialization(bg, N_frames, cap, count)


def change_detection(video_path, bg, threshold):
    ftime = True
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
        #gray = cv2.GaussianBlur(gray, (7, 7), 1)

        #hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        #max = (np.max(hist))
        #peak = np.where(hist == max)
        #peak = peak[0]

        #if 0 <= peak < 50:
         #   r = 4
        #elif 51 <= peak < 102:
         #   r = 2
        #elif 103 <= peak < 153:
         #   r = 0.8
        #elif 154 <= peak < 205:
         #   r = 0.6
        #elif 206 <= peak < 256:
         #   r = 0.4
        #else:
         #   r = 1

        #gray1 = exponential_operator(gray, r)
        #bg1 = exponential_operator(bg, r)

        gray2 = cv2.GaussianBlur(gray, (15, 15), 0)
        bg2 = cv2.GaussianBlur(bg, (15, 15), 0)


        cv2.imshow('stretch', gray2)
        #cv2.imshow('stretchbg', bg1)

        #gray = linear_stretching(np.copy(gray), 130, 0)
        #bg = linear_stretching(np.copy(bg),130,0)

    ### mask with blurred stretched images
        mask = distance(gray2, bg2) > threshold
        mask = mask.astype(np.uint8) * 255
        cv2.imshow('mask', mask)


        open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel,iterations=1)
        dilate = cv2.dilate(open, kernel, iterations=15)

        cv2.imshow('morph', dilate)


        copy_frame = gray.astype(np.uint8)
        cols, rows = copy_frame.shape
        brightness = np.sum(copy_frame) / (255 * cols * rows)
        cv2.imshow('a',copy_frame)
        copy_bg = bg.astype(np.uint8)
        bgbrightness = np.sum(copy_bg) / (255 * cols * rows)
        alpha = brightness / bgbrightness
        new_frame = cv2.convertScaleAbs(copy_frame, alpha = alpha, beta=255 * (1 - alpha))
        new_frame[np.logical_not(dilate)] = np.asarray([255])
        copy_bg[np.logical_not(dilate)] = np.asarray([255])

        hist, bins = np.histogram(copy_bg.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        new=cdf[new_frame]
        newbg=cdf[copy_bg]
        cv2.imshow('new',new)
        cv2.imshow('newbg',newbg)



    ### mask stretched
        mask1 = (distance(new, newbg) > 10)
        mask1 = mask1.astype(np.uint8) * 255
        cv2.imshow('second mask', mask1)
        blur = cv2.GaussianBlur(mask1, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        close1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow('second morph', close1)

        edges = frame.astype(np.uint8)
        edges[np.logical_not(close1)] = np.asarray([-255])

        #find edges and use as a mask for floodfill
        edges = auto_canny(edges)
        mask1 = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        #cv2.imshow('a0', edges)
        final2 = np.copy(close1)
        cv2.floodFill(final2, mask1, (0, 0), 255);
        #cv2.imshow('floodfill1', final2)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(final2)
        #im_floodfill_inv = cv2.dilate(im_floodfill_inv, kernel, iterations=3)
        # Combine the two images to get the foreground.
        cv2.imshow('floodfill2', im_floodfill_inv)
        out = close1 | im_floodfill_inv
        cv2.imshow('floodfill', out)


        hist, bins = np.histogram(out.flatten(), 256, [0, 256])

        #print(compare)


        #find contours
        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        shift2 = np.zeros(out.shape, np.uint8)
        shift1 = np.zeros(out.shape, np.uint8)

        for i, cnt in enumerate(contours):
             #person detector
             if 20000>cv2.contourArea(cnt)>5000:
                 area = cv2.contourArea(cnt)
                 x, y, w, h = cv2.boundingRect(cnt)
                 rect_area = w * h
                 extent = float(area) / rect_area
                 #print(extent)
                 #draw person in blue
                 cv2.drawContours(frame, contours,i,[255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            if cv2.contourArea(contours[j]) < 300 or cv2.contourArea(contours[j]) > 3000:
                continue
            elif skip_background(contours, frame, out , shift1, shift2, j, 0.5) == True:
                #draw false object in red
                cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
            else :
                #draw true objects in green
                cv2.drawContours(frame, contours, j,[0, 255, 0], -1)


        cv2.imshow('contours', frame)

        blob_count = len(contours)

        if ftime == False:

            #if hist[255] > 1.2 * prevhist:
                #cv2.accumulateWeighted(gray, bg, 0.01)
                #print('change_updated ')


            if blob_count < 1:
                #cv2.accumulateWeighted(gray, bg, 0.1)
                print('selective updated')

        ftime = False
        prevhist = hist[255]

        time.sleep(0.02)

        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr)
