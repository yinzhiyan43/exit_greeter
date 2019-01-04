# coding=utf-8

import cv2
import numpy as np

def calcBuleRate(image, left0, left1, right0, right1):
    hsvImage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsvSplit = cv2.split(hsvImage)
    cv2.equalizeHist(hsvSplit[2],hsvSplit[2])
    cv2.merge(hsvSplit,hsvImage)
    thresholded = cv2.inRange(hsvImage,np.array([92,79,25]),np.array([140,255,255]))

    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))

    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,element)

    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_CLOSE,element)

    rateValue = 0.4
    if (left1[0] - left0[0]) < 50:
      if (left0[0]-25) > 0:
        left0[0] = left0[0]-25
      if (left1[0]+25) < image.shape[1]:
        left1[0] = left1[0]+25
      if (right0[0]+25) < image.shape[1]:
        right0[0] = right0[0]+25
      if (right1[0]-25) > 0:
        right1[0] = right1[0]-25
      rateValue = 0.2
    box = np.array([[left0,left1, right0, right1]], dtype = np.int32)
    maskImage = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.polylines(maskImage, box, 1, 255)
    totalArea = calcTotalArea(cv2.fillPoly(maskImage, box, 255));

    blueArea = calcBlueArea(cv2.bitwise_and(thresholded, thresholded, mask=maskImage))
    #cv2.imshow("maskImage",cv2.bitwise_and(thresholded, thresholded, mask=maskImage))
    #cv2.waitKey(0)
    mleft0=[(left0[0]*2/3)+(left1[0]/3),(left0[1]*2/3)+(left1[1]/3)]
    mleft1=[(left0[0]*1/3)+(left1[0]*2/3),(left0[1]*1/3)+(left1[1]*2/3)]
    mright1=[(right0[0]*1/3)+(right1[0]*2/3),(right0[1]*1/3)+(right1[1]*2/3)]
    mright0=[(right0[0]*2/3)+(right1[0]/3),(right0[1]*2/3)+(right1[1]/3)]
    mblueArea = 0
    mtotalArea = 0
    if (totalArea == 0):
        return False,0
    if (blueArea/totalArea) >= rateValue:
        return True,blueArea/totalArea
    if (blueArea/totalArea) < rateValue and rateValue == 0.4:
        mbox = np.array([[mleft0,mleft1, mright0, mright1]], dtype = np.int32)
        mmaskImage = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.polylines(mmaskImage, mbox, 1, 255)
        mtotalArea = calcTotalArea(cv2.fillPoly(mmaskImage, mbox, 255))
        mblueArea = calcBlueArea(cv2.bitwise_and(thresholded, thresholded, mask=mmaskImage))

    print(((blueArea-mblueArea)/(totalArea-mtotalArea)))
    return ((blueArea-mblueArea)/(totalArea-mtotalArea)) >= rateValue,(blueArea-mblueArea)/(totalArea-mtotalArea)

def calcTotalArea(maskImage):
    _, binary = cv2.threshold(maskImage, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    _,contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    area = cv2.contourArea(contours[0])
    return area

def calcBlueArea(maskedImage):
    _, binary = cv2.threshold(maskedImage, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    maxArea  = 0
    blueArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > maxArea:
            maxArea=area
        blueArea += area
    return blueArea

