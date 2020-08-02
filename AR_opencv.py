import cv2
import numpy as np

cap= cv2.VideoCapture(-1)
imgTarget= cv2.imread('TargetImage.jpg',1)
detection=False
framecounter=0
myvid =cv2.VideoCapture('ironman.mp4')
_, myvideo=myvid.read()

ht,wt,ct =imgTarget.shape# we want to get the size of the target to lay the image on top of the photo
imgvedio= cv2.resize(myvideo, (wt, ht)) # resize video to the width and height of target image

orb= cv2.ORB_create(nfeatures=1000) #detector for keypoints and features. Other similar detectors are sift and surf
kp1,des1=orb.detectAndCompute(imgTarget,None)
# imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)
# once we detected the keypoints successfully then we can use the same to detect in the video
while True:
    _,imgcam= cap.read()
    imgaug=imgcam.copy()
    kp2, des2 = orb.detectAndCompute(imgcam, None)
#     after identifying the key points  we will use the brutforce matcher to match this in real time
    bf= cv2.BFMatcher()
    matches= bf.knnMatch(des1,des2,k=2) #query means first reference pic and train means second pic(webcam img)
    good=[]
    if detection==False:
        myvid.set(cv2.CAP_PROP_POS_FRAMES,0) # setting frames to 0
        framecounter=0
    else:
        if framecounter==myvid.get(cv2.CAP_PROP_FRAME_COUNT): # if video has finished execution start over
            myvid.set(cv2.CAP_PROP_POS_FRAMES, 0) #Resting the frame to beginning of the video
            framecounter=0
        _, myvideo=myvid.read()
        # print('inside vedio loop ')
        myvideo=cv2.resize(myvideo, (wt, ht))
    for m,n in matches: # m , n is there because we gave k=2 in knnmatch
        if m.distance < .75*n.distance:
            good.append(m)
    # print(len(good))
    # imgFeatures =cv2.drawMatches(imgTarget,kp1,imgcam,kp2,good,None,flags=2)

    if len(good) >20:
        # print(detection,framecounter)
        detection=True
        srcpts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix,mask= cv2.findHomography(srcpts,dstpts,cv2.RANSAC,5) #to draw the bounting box based on the keypoints
        # print(matrix)
        pts=np.float32([[0,0],[0,ht],[wt,ht],[wt,0]]).reshape(-1,1,2)
        dst= cv2.perspectiveTransform(pts,matrix)
        img2 =cv2.polylines(imgcam,[np.int32(dst)],True,(255,0,255),3,cv2.LINE_AA)
        imgWrap= cv2.warpPerspective(myvideo, matrix, (imgcam.shape[1], imgcam.shape[0])) #changing the perspective of video based on webcam target image

        masknew= np.zeros((imgcam.shape[0],imgcam.shape[1]),np.uint8) #create a black screen same size of cam input
        cv2.fillPoly(masknew,[np.int32(dst)],(255,255,255)) # make the rectangle white
        maskinverse=cv2.bitwise_not(masknew) #flip the mask make everything white and required portion black
        imgaug=cv2.bitwise_and(imgaug,imgaug,mask=maskinverse) # make that mask on top of our camera screen
        imgaug= cv2.bitwise_or(imgWrap,imgaug)
    cv2.imshow('VRoutput',imgaug)
    k = cv2.waitKey(1) & 0xFF
    framecounter+=1
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


