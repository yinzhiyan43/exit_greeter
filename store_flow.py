# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import pickle
import time
import os
import numpy as np
import cmath
from opencv import *
from sys import platform

# Remember to add your installation path here
# Option a
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    sys.path.append(dir_path + '/../../python/openpose/')
else:
    sys.path.append('../../python')
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
def pross():

    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = dir_path + "/../../../models/"
    # Construct OpenPose object allocates GPU memory
    openpose = OpenPose(params)

    #video_path = "/woody/software/source/openpose/examples/media/video.avi"
    video_path = "/home/woody/tmp/openpose/test.mp4"
    #video_path = "/home/woody/tmp/openpose/video/4804_exit_overvie.mp4"
    video = cv2.VideoCapture()

    if not video.open(video_path):
       print("can not open the video")
       exit(1)
    index = 1
    count = 0
    f_count = 1
    imageArray = {}
    continuityArray = []
    up_imageArray = []
    start = time.time()
    while True:
       _, frame = video.read()
       if frame is None:
          break
       if f_count % 15 == 0:
            st = time.time()
            output_image = frame
            keypoints,scores,output_image = openpose.forward(frame, True)
            print("openpose>>>" + str(time.time() - st))

            image = output_image
            flag = False
            continuity = False
            key = "image_info_" + str(index)
            #cv2.imshow("image",image)
            #cv2.waitKey(0)
            #print("start ===============" + key)

            for keypoint in keypoints:
                if calcBuleRate(image,[keypoint[12][0],keypoint[12][1]], [keypoint[9][0],keypoint[9][1]], [keypoint[2][0],keypoint[2][1]],[keypoint[5][0],keypoint[5][1]]) == False:
                   continue
                x1 = keypoint[2][0]
                y1 = keypoint[2][1]
                x2 = keypoint[9][0]
                y2 = keypoint[9][1]
                x3 = keypoint[10][0]
                y3 = keypoint[10][1]
                r_result,r_flag = calcHipAngle(x1,y1,x2,y2,x3,y3)
                x1 = keypoint[5][0]
                y1 = keypoint[5][1]
                x2 = keypoint[12][0]
                y2 = keypoint[12][1]
                x3 = keypoint[13][0]
                y3 = keypoint[13][1]
                l_result,l_flag = calcHipAngle(x1,y1,x2,y2,x3,y3)

                x1 = keypoint[9][0]
                y1 = keypoint[9][1]
                x2 = keypoint[10][0]
                y2 = keypoint[10][1]
                x3 = keypoint[11][0]
                y3 = keypoint[11][1]
                ra_result,ra_flag = calcKneeAngle(x1,y1,x2,y2,x3,y3)
                ra_len_result,ra_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
                x1 = keypoint[12][0]
                y1 = keypoint[12][1]
                x2 = keypoint[13][0]
                y2 = keypoint[13][1]
                x3 = keypoint[14][0]
                y3 = keypoint[14][1]
                la_result,la_flag= calcKneeAngle(x1,y1,x2,y2,x3,y3)
                la_len_result,la_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
                if  ra_flag and la_flag :
                    flag = True
                if  ra_len_flag and la_len_flag :
                    flag = True
                if (r_flag or l_flag) and (abs(r_result - l_result) <= 30) and (ra_result or la_result):
                    flag = True
                if la_result >= 170 or ra_result >= 170:
                    flag = False

                if (la_len_result >= 0.9 and la_len_result <= 1.09) or (ra_len_result >= 0.9 and ra_len_result <= 1.09):
                    flag = False
                if flag:
                    continuityArray.append(keypoint[8])
                    continuity = True

            if continuity:
                v_flag = False
                #print(">>>>>" + str(up_imageArray))
                if len(up_imageArray) >= 1:
                    if len(continuityArray) >= 1:
                        for u_index in range(len(up_imageArray)):
                            for c_index in range(len(continuityArray)):
                                midHip2 = (up_imageArray[u_index][0]-continuityArray[c_index][0])*(up_imageArray[u_index][0]-continuityArray[c_index][0])+(up_imageArray[u_index][1]-continuityArray[c_index][1])*(up_imageArray[u_index][1]-continuityArray[c_index][1])
                                midHip = cmath.sqrt(midHip2)
                                f_midHip = cmath.sqrt(0.09 * (up_imageArray[u_index][0] + up_imageArray[u_index][1]) * (up_imageArray[u_index][0] + up_imageArray[u_index][1]))
                                if midHip.real < f_midHip.real:
                                    v_flag = True
                                    break
                                #print(str(key) + ">>>" + str(midHip.real) + "," + str(f_midHip.real))
                            if v_flag:
                                break
                up_imageArray = continuityArray
                continuityArray = []
                if count < 1:
                    v_flag = True
                if v_flag:
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(key))
                    cv2.imwrite(save_path, image)
                    if 'index' in imageArray.keys():
                        if imageArray['start'] >= count -10:
                            if imageArray['index'] == count -1:
                                imageArray['start'] = count
                                imageArray['count'] = imageArray['count'] + 1

                            imageArray['index'] = count
                        else:
                            save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                            if imageArray['count'] > 1:
                                cv2.imwrite(save_path, imageArray["image"])
                            imageArray = {}
                    else:
                        imageArray['index'] = count
                        imageArray['start'] = count
                        imageArray['count'] = 0
                        imageArray['image'] = output_image
                        imageArray['key'] = key
                else:
                    if 'index' in imageArray.keys():
                        imageArray['index'] = count
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/hh", str(key))
                    cv2.imwrite(save_path, image)

            else:
                if 'index' in imageArray.keys():
                    if imageArray['start'] < count -10:
                        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                        if imageArray['count'] > 1:
                            cv2.imwrite(save_path, imageArray["image"])
                        imageArray = {}
                else:
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/wu", str(key))
                    cv2.imwrite(save_path, image)
                    up_imageArray = []
                    continuityArray = []
            count = count + 1
            print("end ===============" + key)
            print(str(count) + ",totail>>>" + str(time.time() - start))
            index += 1
       f_count = f_count + 1
    if 'index' in imageArray.keys():
        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
        if imageArray['count'] > 1:
            cv2.imwrite(save_path, imageArray["image"])
    print(">>>" + str(time.time() - start))
    video.release()
    print("Totally save {:d} pics".format(index - 1))


def calcHipAngle(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 0,False
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    pos = (a2+b2-c2)/(2*a*b)
    angle = cmath.acos(pos)
    realangle = angle*180/cmath.pi
    #print("calcHipAngle:" + str(realangle.real));
    if (realangle.real >= 30 and realangle.real <= 140) :
        return realangle.real,True
    else:
        return realangle.real,False

def calcKneeAngle(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 0,False
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    pos = (a2+b2-c2)/(2*a*b)
    angle = cmath.acos(pos)
    realangle = angle*180/cmath.pi
    #print("calcKneeAngle:" + str(realangle.real))
    if (realangle.real <= 140) :
        return realangle.real,True
    else:
        return realangle.real,False

def calcLenRate(x1,y1,x2,y2,x3,y3):
    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0 or x3 == 0 or y3 == 0:
        return 0,False
    a2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    b2 = (x3-x2)*(x3-x2)+(y3-y2)*(y3-y2)
    c2 = (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
    d2 = (y1-y2)*(y1-y2)
    a = cmath.sqrt(a2)
    b = cmath.sqrt(b2)
    c = cmath.sqrt(c2)
    d = cmath.sqrt(d2)
    result = max(a.real, b.real)/d.real
    #print("calcLenRate:" + str(result))
    if result >= 0.8 and result <= 1.2 :
        return result,False
    else:
        return result,True


if __name__ == '__main__':
    pross()
