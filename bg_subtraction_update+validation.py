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
        #gray = cv2.GaussianBlur(gray, (7, 7), 1)
        cv2.imshow('blur',gray)

        #hist1, bins = np.histogram(gray.flatten(), 256, [0, 256])
        #eq=pfm(hist1)
        #gray1=eq[gray]
        #bg = bg.astype(np.uint8)
        #hist2, bins = np.histogram(bg.flatten(), 256, [0, 256])
        #eq2=pfm(hist2)
        #bg1=eq2[bg]

        #set exponential stretching from histogram peak
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        max = (np.max(hist))
        peak = np.where(hist == max)
        peak = peak[0]

        if 0 <= peak < 50:
            r= 4
        elif 51 <= peak < 102:
            r= 2
        elif 103 <= peak < 153:
            r= 0.8
        elif 154 <= peak < 205:
            r= 0.6
        elif 206 <= peak < 256:
            r= 0.4
        else:
            r=1

        gray1=exponential_operator(gray,r)
        bg1=exponential_operator(bg,r)

        #cv2.imshow('stretch', gray1)
        #cv2.imshow('stretchbg', bg1)

        #gray = linear_stretching(np.copy(gray), 130, 0)
        #bg = linear_stretching(np.copy(bg),130,0)

        mask = distance(gray1, bg1) > threshold
        mask = mask.astype(np.uint8) * 255
        cv2.imshow('mask', mask)


        open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel,iterations=3)
        final = cv2.dilate(close, kernel, iterations=3)

        cv2.imshow('morph', final)



        edges = frame.astype(np.uint8)
        edges[np.logical_not(final)] = np.asarray([-255])

        #find edges and use as a mask for floodfill
        edges = cv2.Canny(edges,100,200)
        mask1 = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        #cv2.imshow('a0', mask1)
        final2 = np.copy(final)
        cv2.floodFill(final2, mask1, (0, 0), 255);
        #cv2.imshow('floodfill1', final2)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(final2)
        im_floodfill_inv = cv2.dilate(im_floodfill_inv, kernel, iterations=3)
        # Combine the two images to get the foreground.
        cv2.imshow('floodfill2', im_floodfill_inv)
        out = final | im_floodfill_inv
        cv2.imshow('floodfill', out)


        hist, bins = np.histogram(out.flatten(), 256, [0, 256])

        #print(compare)
        # update background when ligth changes

        if ftime == False:

            if hist[255] > 1.3 * prevhist:
                cv2.accumulateWeighted(gray, bg, 0.05)
                print('change_updated ')

            elif hist[255] < 0.3 * prevhist:
                cv2.accumulateWeighted(gray, bg, 0.3)
                print('selective updated')

        ftime = False
        prevhist = hist[255]


        #find contours
        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        shift2 = np.zeros(out.shape, np.uint8)
        shift1 = np.zeros(out.shape, np.uint8)

        for i, cnt in enumerate(contours):
             #person detector
             if cv2.contourArea(cnt)>5000:
                 #draw person in blue
                cv2.drawContours(frame, contours,i,[255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            if cv2.contourArea(contours[j]) < 500 or cv2.contourArea(contours[j]) > 2000:
                continue
            elif skip_background(contours, frame, out , shift1, shift2, j, 0.5) == True:
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
