import numpy as np
import cv2
import time


def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c  # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()


def hist_match(original, specified):
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles * 255)
    temp = np.around(t_quantiles * 255)

    # Map the rounded values
    b = []
    for data in sour[:]:
        b.append(find_nearest_above(temp, data))
    b = np.array(b, dtype='uint8')

    return b[bin_idx].reshape(oldshape)


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
    return [maskbool]


def threeframedifference(frame, prev1, prev2, distance_type, threshold):
    if distance_type == "L1":
        [maskbool, mask1] = twoframedifference(frame, prev1, distance_type, threshold)
        if len(prev2) != 0:
            [maskbool1, mask2] = twoframedifference(prev1, prev2, distance_type, threshold)
    mask = np.logical_and(maskbool, maskbool1)
    prev1[np.logical_not(mask)] = np.array([255, 255, 255])
    mask = mask.astype(np.uint8) * 255


def pfm(hist):
    total_pixel = np.sum(hist)
    pfm = []
    for i in range(256):
        pfm_i = np.sum(hist[:i]) / total_pixel
        pfm.append(pfm_i)
    return np.asarray(pfm)


# Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean




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
    #cv2.imshow('g1', frame)

    bg2 = np.copy(prev_bg)
    bg3 = np.copy(prev_bg)
    prev_bg = prev_bg.astype(np.uint8)
    prev_bg[np.logical_not(closing)] = np.asarray(0)
    #cv2.imshow('g2', prev_bg)
    bg3[closing == 255] = np.asarray(0)
    #cv2.imshow('bg3', bg3)
    bg2 = (1 - alfa) * prev_bg + alfa * frame + bg3
    bg1 = np.copy(bg2)
    #cv2.imshow('g4', bg2.astype(np.uint8))
    # cv2.imshow('g3', bg1.astype(np.uint8))
    # bg1=cv2.accumulateWeighted(bg, prev_bg, 0.05)
    # bg1=cv2.GaussianBlur(bg1, (5, 5), 0)
    return bg1


def background_update(bg1, bg, prev_bg, alfa):
    bg1 = (1 - alfa) * prev_bg + alfa * bg
    # bg1=cv2.accumulateWeighted(bg, prev_bg, 0.05)
    # bg1=cv2.GaussianBlur(bg1, (5, 5), 0)
    return bg1


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


# def b_up(bg2,new,bg,prev,alfa):
# new2=new
# bg2.append(new)
# bg2.append(bg)
# bg2.append(prev)
# new1=interpolation(bg2)
# new2 = (1 - alfa) * new1 + alfa * new2
# return new2
###Define change detection parameters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 30
distance = L2
bg = []
b = []
bg1 = []
bg2 = []
frame = []
N_frames = 30  # then refresh


# denoising_kernel = np.array([
#            [1,2,1],
#            [2,4,2],
#            [1,2,1]])/16

def sobel(img):
    dst1 = cv2.Sobel(img, -1, 1, 0, 3)
    dst2 = cv2.Sobel(img, -1, 0, 1, 3)
    sob = np.maximum(dst1, dst2)
    return sob


k_size = 5
mean_kernel = np.ones([k_size, k_size]) / (k_size ** 2)

# blob detector parameters

cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
[b, bg, count] = background_initialization(bg, N_frames, cap, count)


# fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def change_detection(video_path,bg, threshold, frame, b):
    # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    # bg_inter1=[]
    # prevhist2 = 0
    cond = False
    ftime = True
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
        sgray =np.copy(gray)
        sbg = np.copy(gray)

        if ftime==True:
            pgray=np.copy(sgray)
            ftime=False
           # bg=sgray.astype(np.uint8)
            selective_bg=[sbg]
            idx=0

        elif idx<100:

            smask = (distance(sgray, pgray) > 5)
            smask=smask.astype(np.uint8) * 255
            cv2.imshow('premask',smask)
            pgray=sgray
            sopen=cv2.morphologyEx(smask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                      iterations = 2)
            sdilate=cv2.dilate(sopen, None, iterations=5)
            sinv = 255 - sdilate
            sbg[np.logical_not(sinv)] = np.asarray(0)
            sbg1=np.copy(sbg)
            cv2.imshow('prem', sbg1)

            if idx % 2 == 0:
                selective_bg.append(sbg)
            selective = np.stack(selective_bg, axis=0)
            selective = interpolation(selective, axis=0)
            selective = selective.astype(np.uint8)
            sbg=selective
                #selective_bg2.append(selective)
                #selective2 = np.stack(selective_bg2, axis=0)
                #selective2 = interpolation(selective2, axis=0)
                #selective2 = selective2.astype(np.uint8)
               # alfa=0.1
               # new_bg = np.zeros(final.shape, np.uint8)
            # new_bg = (1 - alfa) * selective_bg + alfa * selective_bg
               # asbg=cv2.multiply(alfa, selective_bg[0])
               # aselective=cv2.multiply((1 - alfa), selective_bg[1])
               # aselective=aselective.astype(np.uint8)
               # cv2.add(asbg,aselective,new_bg)
               # selective_bg.pop(-1)
               # cv2.imshow('premask', new_bg)


            idx += 1
            print(idx)
            cv2.imshow('selective', sbg)


        # bg=linear_stretching(np.copy(bg), 255,170)
        # bg2=bg.astype(np.uint8)
        # bg2[gray4<255]=np.asarray(0)
        #cv2.imshow('gray34', gray3)
        # cv2.imshow('gray2', bg.astype(np.uint8))
        #cv2.imshow('bg', bg)
        # cv2.imshow('gray23', bg2)
        # Compute background suptraction
        bg=sbg
        gray0 = cv2.GaussianBlur(gray, (15, 15), 0)
        bg0 = cv2.GaussianBlur(bg, (15, 15), 0)
        mask = (distance(gray0, bg0) > 10)

        mask = mask.astype(np.uint8) * 255
        # mask= fgbg.apply(gray)
        cv2.imshow('mask', mask)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        #cv2.imshow('Blur', blur)
        # blur2 = cv2.filter2D(blur,-1,denoising_kernel)
        # blur2=cv2.fastNlMeansDenoising(blur)
        # cv2.imshow('Blur2', blur2)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)





            #ssbg[inv == 255] = np.asarray(0)
            #cv2.imshow('prem3', ssbg)
            #cv2.accumulateWeighted(sbg, selective_bg, 0.2)
            #alfa=0.1
            #new_bg = np.zeros(final.shape, np.uint8)
            #new_bg = (1 - alfa) * selective_bg + alfa * selective_bg
            #sbg=cv2.multiply(alfa, selective_bg[0])
            #selective=cv2.multiply((1 - alfa), selective_bg[1])
            #cv2.add(sbg,selective,new_bg)
            #selective_bg.pop(-1)
            #cv2.imshow('premask', new_bg)

        #cv2.imshow('thresh', thresh)
        # im_out = thresh | im_floodfill_inv
        # cv2.imshow('combine', im_out)
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                   iterations=3)
        # cv2.imshow('opening', opening)
        dilated = cv2.dilate(opening, None, iterations=11)
        # dilated2 = cv2.bitwise_not(dilated)
        # clos = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        #cv2.imshow("clos", dilated)
        closing = dilated
        inv_closing = 255 - closing
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 5)
        cv2.imshow('closing', closing)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        # edges = gray.astype(np.uint8)
        mask[np.logical_not(closing)] = np.asarray(0)
        blur[np.logical_not(closing)] = np.asarray(0)

        mask6 = (distance(gray, bg) > 10)

        mask6 = mask6.astype(np.uint8) * 255
        cv2.imshow('second mask', mask6)
        mask6[np.logical_not(closing)] = np.asarray(0)
        blur6 = cv2.GaussianBlur(mask6, (5, 5), 3)
        ret6, thresh6 = cv2.threshold(blur6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('mask6', mask6)
        # cv2.imshow('blur6', blur6)
        #cv2.imshow('thresh6', thresh6)

        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                    iterations=2)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                    iterations=2)
        #
        # cv2.imshow('e1', edges2)
        #cv2.imshow('e2', opening2)
        #cv2.imshow('e22', closing2)
        f = gray.astype(np.uint8)
        f[np.logical_not(closing2)] = np.asarray(0)
        # cv2.imshow('e3', f)

        out = closing2
        # im[np.logical_not(closing2)] = np.asarray(0)
        # cv2.imshow('im', im)
        im2 = gray.copy()
        closing4 = cv2.dilate(closing2, None, iterations=2)
        #cv2.imshow('closing4', closing4)
        closing3 = 255 - closing4
        im2[np.logical_not(closing3)] = np.asarray(0)
        #cv2.imshow('im2', im2)


        hist, bins = np.histogram(thresh.flatten(), 256, [0, 256])


        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        final = out
        blob_count = len(contours)
         # cv2.drawContours(im_keypoints2, [cnt], 0, (255, 255, 255), 3)

        # cv2.imshow('contours', frame)
        if ((blob_count < 1) and (hist[255] < 0.8 * prevhist)):
            bg = selective_background_update(bg1, gray, bg, 0.6, closing3)
            # bg = background_update(bg1, im2, bg, 0.1)
            print('background update')

        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
            # person detector
            if cv2.contourArea(cnt) > 6000:
                # draw person in blue
                cv2.drawContours(frame, contours, i, [255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            if cv2.contourArea(contours[j]) < 100 or cv2.contourArea(contours[j]) > 2000:
                continue
            elif skip_background(contours, frame, final, shift1, shift2, j, 0.9) == True:
                # draw false object in red
                cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
            else:
                # draw true objects in green
                cv2.drawContours(frame, contours, j, [0, 255, 0], -1)

        cv2.imshow('contours', frame)
        time.sleep(0.02)
        # if (cond==True):
        #     cond2 = True
        if (cond == True and hist[255] > 1.1 * prevhist):

            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
            # bg = background_update(bg1, l4, bg, 0.2, inv_closing)
            print('change_updated')

        # elif (cond==True and hist[255] == 0):
        #     bg = background_update(bg1, gray, bg, 0.2)
        # bg = gray + 0
        #     print('change_updated6')
        prevhist = hist[255]
        prevhist2 = hist[0]
        cond = True
        # prevfr=gray3-im
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi',bg, thr, frame, b)