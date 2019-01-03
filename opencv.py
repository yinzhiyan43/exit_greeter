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
        return False
    if (blueArea/totalArea) >= rateValue:
        return True
    if (blueArea/totalArea) < rateValue and rateValue == 0.4:
        mbox = np.array([[mleft0,mleft1, mright0, mright1]], dtype = np.int32)
        mmaskImage = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.polylines(mmaskImage, mbox, 1, 255)
        mtotalArea = calcTotalArea(cv2.fillPoly(mmaskImage, mbox, 255))
        mblueArea = calcBlueArea(cv2.bitwise_and(thresholded, thresholded, mask=mmaskImage))

    print(((blueArea-mblueArea)/(totalArea-mtotalArea)))
    return ((blueArea-mblueArea)/(totalArea-mtotalArea)) >= rateValue

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

#if __name__=='__main__':

    #image = cv2.imread("/home/woody/sams/p/1.jpg")
    #print(calcBuleRate(image,[129,447], [304,448], [303,237],[57,264]))

    #image = cv2.imread("/home/woody/sams/data/5.jpg")
    #print(calcBuleRate(image,[869,903], [939,903],[968,702],[837,705]))
    #print(calcBuleRate(image,[1629,907], [1706,907],[1719,730],[1596,726]))
    #print(calcBuleRate(image,[373,931], [467,927],[492,701],[336,701]))
    #print(calcBuleRate(image,[1296,865], [1390,861],[1403,640],[1259,640]))
    #print(calcBuleRate(image,[2285,833], [2379,825],[2379,644],[2244,648]))
    #print(calcBuleRate(image,[1953,894], [2043,890],[2043,697],[1916,697]))

    #image = cv2.imread("/home/woody/sams/data/3.jpg")
    #cv2.imshow("image",image);
    #cv2.waitKey(0)
    #print(calcBuleRate(image,[339,169], [356,178],[378,150],[353,136]))
    #print(calcBuleRate(image,[178,166], [188,166],[192,139],[176,140]))
    #print(calcBuleRate(image,[236,164], [248,164],[252,134],[234,134]))
    #print(calcBuleRate(image,[96,220], [107,216],[97,172],[87,177]))
    #print(calcBuleRate(image,[96,220], [107,216],[87,177],[97,172]))


