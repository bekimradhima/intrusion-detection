import numpy as np
import cv2
#from matplotlib import pyplot as plt
import time 

def L1(img1, img2):
    diff = np.abs(img1-img2)
    print(len(img1.shape))
    if img1.shape[-1] ==  3 and len(img1.shape)==3:
        diff = np.sum(diff, axis=-1)
    return diff

def L2(img1, img2):
    sq_dist = (img1-img2)**2
    if img1.shape[-1] ==  3 and len(img1.shape)==3:
        sq_dist = np.sum(sq_dist,axis=-1)
    diff = np.sqrt(sq_dist)
    return diff

def Linf(F1,F2):
    diff = np.abs(F1-F2)
    if F1.shape[-1] ==  3 and len(F1.shape)==3:
        diff = np.max(diff, axis=-1)
    return diff
   
def twoframedifference(frame, previousframe, distance_type, threshold):
        distance=distance_type(frame,previousframe)
        frame[distance<threshold]=np.array([255,255,255])
        maskbool = distance>threshold
        #print(distance>threshold)
        mask=maskbool.astype(np.uint8)*255
        return [mask]
        #cv2.imshow('frame', frame)
        #cv2.imshow('mask', mask)

def threeframedifference(frame,prev1,prev2, distance_type, threshold):
    if distance_type=="L1":
        [maskbool,mask1]=twoframedifference(frame,prev1, distance_type, threshold)
        if len(prev2)!=0:
            [maskbool1,mask2]=twoframedifference(prev1,prev2, distance_type, threshold)
    #print(maskbool, maskbool1)
    mask=np.logical_and(maskbool, maskbool1)
    prev1[np.logical_not(mask)]=np.array([255,255,255])
    #print(mask)
    mask=mask.astype(np.uint8)*255
    #cv2.imshow('frame', prev1)
    #cv2.imshow('mask', mask)
def pfm(hist):
    total_pixel = np.sum(hist)
    pfm = []
    for i in range(256):
        pfm_i = np.sum(hist[:i]) / total_pixel
        pfm.append(pfm_i)
    return np.asarray(pfm)


# Defining a variable interpolation for mean or median functions
interpolation = np.median # or np.mean
alfa=0.2
def background_initialization(bg,n,cap,count):
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        if ret or not frame is None:
            # Release the Video if ret is false
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #hist,bins = np.histogram(frame.flatten(),256,[0,256])
            #eq_op = pfm(hist)*255
            #frame = eq_op[frame] 
            frame
            if count==0:
                bg.append(frame)
            else :
                bg.append(alfa*frame)
                bg[count]=(1-alfa)*bg[count-1] + bg[count]
            count +=1             
            #print(count)
        else:
            break
    cap.release()
    bg_inter=np.stack(bg, axis=0)
    bg_inter=interpolation(bg_inter,axis=0) 
    cv2.destroyAllWindows()
    return [bg_inter, count]
   
def background_update(bg, prev_bg):
    bg=(1-alfa)*prev_bg + alfa*bg
    return bg

###Define change detection parameters
thr = 65
distance = L2
bg=[]
N_frames=30 #then refresh
#def check_light():
    
def sobel(img):
    dst1=cv2.Sobel(img, -1,  1, 0, 3)
    dst2=cv2.Sobel(img, -1,  0, 1, 3)
    sob= np.maximum(dst1,dst2)
    return sob


# blob detector parameters

personDetectorParameters = cv2.SimpleBlobDetector_Params()
bookDetectorParameters = cv2.SimpleBlobDetector_Params()


# define params for person detection
personDetectorParameters.filterByArea = True
personDetectorParameters.minArea = 5000 #5000
personDetectorParameters.maxArea = 100000
personDetectorParameters.minDistBetweenBlobs = 0
personDetectorParameters.filterByCircularity = False
personDetectorParameters.filterByColor = True
personDetectorParameters.blobColor = 255
personDetectorParameters.filterByConvexity = False
personDetectorParameters.filterByInertia = False


# define params for book detection
bookDetectorParameters.filterByArea = True
bookDetectorParameters.minArea = 1000 #1000
bookDetectorParameters.maxArea = 5000 #5000
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
#computation of the background
[bg, count]=background_initialization(bg,N_frames,cap, count)

# bg=cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY)

def change_detection(video_path, bg, threshold,idx):
   # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break
        # Display the frame
        #if idx < 1:
        #    previous_frames.append(frame.astype(float))
        #    mask = distance(frame, bg) > threshold
        #else:
        #    masks = []
        #    mask = distance(frame, bg) > threshold
        #    masks.append(mask)

         #   if len(previous_frames)>0:
          #      mask1 = distance(previous_frames[0],bg) > threshold
           #     masks.append(mask1)
            #    mask=np.prod(masks, axis=0)

            #previous_frames.pop(0)
            #previous_frames.append(frame.astype(float))

        #idx += 1
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #mask=1-bg[frame]>threshold
        mask = distance(frame, bg) > threshold
        mask = mask.astype(np.uint8)*255
        #cv2.imshow('mask', mask.astype(np.uint8)*255)
        blur=cv2.GaussianBlur(mask.astype(np.uint8)*255,(5,5),0)
        # cv2.imshow('Blur', blur)
        #ret, thresh = cv2.threshold(blur, 100, 255, 0) no difference in the video with otsu but i like japanese
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        # we fill and erode non essential changes in the image first
        #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        #cv2.imshow('opening', opening)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        #cv2.imshow('closing', closing)
        eroded = cv2.erode(thresh, None, iterations=10)
        cv2.imshow('eroded', eroded)
        dilated = cv2.dilate(eroded, None, iterations=10)
        cv2.imshow('dilated', dilated)
        # eroded2 = cv2.erode(dilated, None, iterations=14)
        # cv2.imshow('eroded2', eroded2)
        # then holes are filled with dilation
        # dilated2 = cv2.dilate(eroded, None, iterations=1)
        # cv2.imshow('dilated', dilated2)
        selective=dilated+mask
        cv2.imshow('selective', selective)
        #if (selective==np.zeros(mask.shape)):
        #    bg=background_update(frame, bg)
        keypoints = detector_person.detect(dilated)
       # keypoints = detector_person.detect(closing)
        #if (idx==200):
        #    i=1
        #    j=1
        #    bg=background_update(frame,bg,i,j)
        #    idx=0
        im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints2 = detector_book.detect(dilated)
        #keypoints2 = detector_person.detect(closing)
        im_keypoints2 = cv2.drawKeypoints(im_keypoints, keypoints2, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Video',im_keypoints2)
        #time.sleep(0.02)
        idx +=1
        if cv2.waitKey(1) == ord('q'):
                break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()
    
change_detection('1.avi', bg, thr, idx)
# something something this way comes