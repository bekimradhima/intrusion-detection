import numpy as np
import cv2
import time


def L1(img1, img2):
    diff = np.abs(img1 - img2)
    print(len(img1.shape))
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff


def L2(img1, img2):
    sq_dist = (img1 - img2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff


def Linf(F1, F2):
    diff = np.abs(F1 - F2)
    if F1.shape[-1] == 3 and len(F1.shape) == 3:
        diff = np.max(diff, axis=-1)
    return diff


def twoframedifference(frame, previousframe, distance_type, threshold):
    distance = distance_type(frame, previousframe)
    maskbool = distance > threshold
    mask = maskbool.astype(np.uint8) * 255
    return [mask]


def pfm(hist):
    total_pixel = np.sum(hist)
    pfm = []
    for i in range(256):
        pfm_i = np.sum(hist[:i]) / total_pixel
        pfm.append(pfm_i)
    return np.asarray(pfm)


Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean
alfa = 0.2


def background_initialization(bg, n, cap, count):
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            # Release the Video if ret is false
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hist,bins = np.histogram(frame.flatten(),256,[0,256])
            # eq_op = pfm(hist)*255
            # frame = eq_op[frame]
            frame
            if count == 0:
                bg.append(frame)
            else:
                bg.append(alfa * frame)
                bg[count] = (1 - alfa) * bg[count - 1] + bg[count]
            count += 1
            # print(count)
        else:
            break
    cap.release()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [bg_inter, count]


def background_update(bg, prev_bg):
    bg = (1 - alfa) * prev_bg + alfa * bg
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg


###Define change detection parameters
thr = 25
distance = L2
bg = []
N_frames = 70 # then refresh


# def check_light():

def sobel(img):
    dst1 = cv2.Sobel(img, -1, 1, 0, 3)
    dst2 = cv2.Sobel(img, -1, 0, 1, 3)
    sob = np.maximum(dst1, dst2)
    return sob


# blob detector parameters

personDetectorParameters = cv2.SimpleBlobDetector_Params()
bookDetectorParameters = cv2.SimpleBlobDetector_Params()

# define params for person detection
personDetectorParameters.filterByArea = True
personDetectorParameters.minArea = 5000  # 5000
personDetectorParameters.maxArea = 100000
personDetectorParameters.minDistBetweenBlobs = 0
personDetectorParameters.filterByCircularity = False
personDetectorParameters.filterByColor = True
personDetectorParameters.blobColor = 255
personDetectorParameters.filterByConvexity = False
personDetectorParameters.filterByInertia = False

# define params for book detection
bookDetectorParameters.filterByArea = True
bookDetectorParameters.minArea = 1000  # 1000
bookDetectorParameters.maxArea = 5000  # 5000
bookDetectorParameters.minDistBetweenBlobs = 0
bookDetectorParameters.filterByCircularity = False
bookDetectorParameters.filterByColor = True
bookDetectorParameters.blobColor = 255
bookDetectorParameters.filterByConvexity = False
bookDetectorParameters.filterByInertia = False

detector_person = cv2.SimpleBlobDetector_create(personDetectorParameters)
detector_book = cv2.SimpleBlobDetector_create(bookDetectorParameters)
cap = cv2.VideoCapture('1.avi')
count = 0
idx = 0
# computation of the background
[bg, count] = background_initialization(bg, N_frames, cap, count)
#bg= cv2.GaussianBlur(bg, (5, 5), 0)
oldbg=bg
bg1=bg

#fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)


def change_detection(video_path, bg, threshold, idx):
    # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    while (cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break
        #Convert to grayscale and blur frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        #Compute background suptraction
        mask = (distance(gray, bg) > threshold)
        mask = mask.astype(np.uint8) * 255


        #mask= fgbg.apply(gray)

        cv2.imshow('mask', mask)

        #Erode mask to minimize false changes, blur to shade figures and threshold to get contours
        eroded1 = cv2.erode(mask, None, iterations=3)
        dilated1 = cv2.dilate(eroded1, Rectangular_kernel, iterations=3)
        blur = cv2.GaussianBlur(dilated1, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #Dilate figure and close small gaps
        dilated = cv2.dilate(thresh, Rectangular_kernel, iterations=3)
        final = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, Rectangular_kernel)

        cv2.imshow('closed', final)



        #compute the difference with background using alpha blending
        #bgg = bg.astype(np.uint8)
        #bgg1 = bg.astype(np.uint8)
        #bgg[np.logical_not(closed)] = np.asarray([-255])
        #closed1 = 255 - closed
        #bgg1[np.logical_not(closed1)] = np.asarray([255])
        #bgc = bgg + bgg1
        #tbg = background_update(bgc, bg)
        # bgc=bgg+gray1
        #cv2.imshow('background', tbg.astype(np.uint8))


        #bgmask = distance(alfa*gray+(1-alfa)*bg,bg) > 20
        #bgmask= bgmask.astype(np.uint8) * 255


        #avgbg = cv2.convertScaleAbs(avgbg)
        #cv2.imshow('bgmask',bgmask)
        #print(np.mean(bgmask))

        #selective update when no change detected with background
        #if np.mean(bgmask) < 0.1 :
           # bg = background_update(gray, bg)
           # print('selective update')

        hist, bins = np.histogram(final.flatten(), 256, [0, 256])


        #update background when ligth changes
        if hist[255] > 1.1*prevhist :
            cv2.accumulateWeighted(gray, bg, 0.05)
            print('change_updated')
        prevhist=hist[255]






        #keypoints = detector_person.detect(dilated)
        #im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #keypoints2 = detector_book.detect(dilated)
        #im_keypoints2 = cv2.drawKeypoints(im_keypoints, keypoints2, np.array([]), (255, 0, 0),
        #                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow('Video', im_keypoints2)
        _, contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
            if cv2.contourArea(cnt) < 1000:
                continue
            elif cv2.contourArea(cnt) < 2000:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 1)  # if >0 shows contour
            else:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('contours', frame)
        #cv2.resizeWindow('contours', 500, 500)

        time.sleep(0.02)
        idx += 1
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, idx)
