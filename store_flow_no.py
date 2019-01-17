# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import pickle
import time
import os
import numpy as np
import cmath
import logging
import base64
import json
from PIL import Image
from io import BytesIO
from websocket import Websocket
from sys import platform

# Remember to add your installation path here
# Option a
# Option b
# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
# sys.path.append('/usr/local/python')
dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    from openpose.openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

global servers
servers = []

#设置连续多少帧数有做坐报警
APM_TOTAIL = 8
#设置蓝色分数比例
APM_BULE_SCORE = 0.4
#设置蓝大于色分数比例次数，表示为True
APM_BULE_COUNT = 3

#检查是否连续多少帧数没有坐下数据
E_TOTAIL = 10
#在某段次数超过该数，就判断是坐行为
E_COUNT = 5

#颜色范围设置
#LOW_COLOR_SCALAR = [105, 48, 111]
#HIGH_COLOR_SCALAR = [126, 210, 255]


LOW_COLOR_SCALAR = [92, 79, 25]
HIGH_COLOR_SCALAR = [140, 255, 255]

ws_data_list = []

#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#console = logging.StreamHandler()
console = logging.FileHandler("/home/woody/store.log")
console.setLevel(logging.INFO)
#console.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console)




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
    params["default_model_folder"] = "/woody/software/source/openpose2019/models/"
    # Construct OpenPose object allocates GPU memory

    openpose = OpenPose(params)
    websocket_server = Websocket(9000, servers=[], new_client=new_client, client_left=client_left, message_received=message_received )
    websocket_server.start()
    while True:
        if websocket_server.servers is not None:
            break
    video_path = "/home/woody/tmp/openpose/test1.mp4"
    #video_path = "/woody/software/source/openpose/examples/media/video.avi"


    f_name = "test1.mp4"
    #video_path = "rtsp://172.16.3.26/test"
    #video_path = "/home/woody/tmp/openpose/video/4804_exit_overvie.mp4"
    video = cv2.VideoCapture()
    if not video.open(video_path):
       logger.info("can not open the video")
       exit(1)
    index = 1
    count = 0
    f_count = 1
    imageArray = {}
    continuityArray = []
    up_imageArray = []
    images = []
    start = time.time()
    while True:
       _, frame = video.read()
       if frame is None:
          break
       base64_data = frame2base64(frame)
       data_json = {}
       data_json["type"] = 1
       data_json["image"] = "data:image/jpeg;base64," + base64_data.decode()
       websocket_server.servers[0].send_message_to_all(str(json.dumps(data_json)))
       if f_count % 15 == 0:
            st = time.time()
            output_image = frame
            keypoints, scores = openpose.forward(frame, False)
            logger.info("openpose>>>" + str(time.time() - st))

            image = output_image
            continuity = False
            key = f_name[:-4] + "_" + str(index)

            for keypoint in keypoints:
                flag = False
                bule_flag, bule_totail = calcBuleRate(image, [keypoint[12][0], keypoint[12][1]], [keypoint[9][0],keypoint[9][1]], [keypoint[2][0],keypoint[2][1]],[keypoint[5][0],keypoint[5][1]])
                if bule_flag == False:
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
                ra_result, ra_flag = calcKneeAngle(x1,y1,x2,y2,x3,y3)
                ra_len_result, ra_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
                x1 = keypoint[12][0]
                y1 = keypoint[12][1]
                x2 = keypoint[13][0]
                y2 = keypoint[13][1]
                x3 = keypoint[14][0]
                y3 = keypoint[14][1]
                la_result,la_flag= calcKneeAngle(x1,y1,x2,y2,x3,y3)
                la_len_result,la_len_flag = calcLenRate(x1,y1,x2,y2,x3,y3)
                if ra_flag and la_flag:
                    flag = True
                if ra_len_flag and la_len_flag:
                    flag = True
                if (r_flag or l_flag) and (abs(r_result - l_result) <= 30) and (ra_result or la_result):
                    flag = True
                #if la_result >= 170 or ra_result >= 170:
                #   flag = False

                if (la_len_result >= 0.9 and la_len_result <= 1.09) or (ra_len_result >= 0.9 and ra_len_result <= 1.09):
                    flag = False
                if flag:
                    continuityArray.append(keypoint[8])
                    continuity = True
                logger.info(key + " >>> sat_list >>>>>>>>>" + str(flag) + ">>>>>" + str(continuity))
            if continuity:
                v_flag = False
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
                                #logger.info(str(key) + ">>>" + str(midHip.real) + "," + str(f_midHip.real))
                            if v_flag:
                                break
                up_imageArray = continuityArray
                continuityArray = []
                if count < 1:
                    v_flag = True
                if v_flag:
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/hh", str(key))
                    cv2.imwrite(save_path, image)
                    if 'index' in imageArray.keys():
                        if imageArray['start'] >= count - E_TOTAIL:
                            if imageArray['index'] == count - 1:
                                imageArray['start'] = count
                                imageArray['count'] = imageArray['count'] + 1
                                if imageArray['count'] >= APM_TOTAIL:
                                    if imageArray['apm'] == False:
                                        imageArray['apm'] = True
                                        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                                        cv2.imwrite(save_path, imageArray["image"])

                                        base64_data = frame2base64(imageArray["image"])
                                        data_json = {}
                                        data_json["type"] = 2
                                        data_json["image"] = "data:image/jpeg;base64," + base64_data.decode()
                                        websocket_server.servers[0].send_message_to_all(str(json.dumps(data_json)))
                                        ws_data_list.append(data_json)

                                        image_data = {}
                                        image_data["key"] = key
                                        image_data["image"] = image
                                        images.append(image_data)
                                        for image_info in images:
                                            save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(image_info['key']))
                                            cv2.imwrite(save_path, image_info["image"])
                                        images = []

                            imageArray['index'] = count
                            if imageArray['apm'] == False:
                                image_data = {}
                                image_data["key"] = key
                                image_data["image"] = image
                                images.append(image_data)

                        else:
                            save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                            if imageArray['apm'] == False:
                                if imageArray['count'] >= APM_TOTAIL:
                                    imageArray['apm'] = True
                                    cv2.imwrite(save_path, imageArray["image"])
                                    base64_data = frame2base64(imageArray["image"])
                                    data_json = {}
                                    data_json["type"] = 2
                                    data_json["image"] = "data:image/jpeg;base64," + base64_data.decode()
                                    websocket_server.servers[0].send_message_to_all(str(json.dumps(data_json)))
                                    ws_data_list.append(data_json)

                                    image_data = {}
                                    image_data["key"] = key
                                    image_data["image"] = image
                                    images.append(image_data)
                                    for image_info in images:
                                        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(image_info['key']))
                                        cv2.imwrite(save_path, image_info["image"])
                                    images = []
                            imageArray = {}
                            images = []

                    else:
                        imageArray['index'] = count
                        imageArray['start'] = count
                        imageArray['count'] = 0
                        imageArray['apm'] = False
                        imageArray['image'] = output_image
                        imageArray['key'] = key
                else:
                    if 'index' in imageArray.keys():
                        imageArray['index'] = count
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/hh", str(key))
                    cv2.imwrite(save_path, image)

            else:
                if 'index' in imageArray.keys():
                    if imageArray['start'] < count - E_TOTAIL:
                        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
                        if imageArray['apm'] == False:
                            if imageArray['count'] >= APM_TOTAIL:
                                imageArray['apm'] = True
                                cv2.imwrite(save_path, imageArray["image"])
                                base64_data = frame2base64(imageArray["image"])
                                data_json = {}
                                data_json["type"] = 2
                                data_json["image"] = "data:image/jpeg;base64," + base64_data.decode()
                                websocket_server.servers[0].send_message_to_all(str(json.dumps(data_json)))
                                ws_data_list.append(data_json)

                                for image_info in images:
                                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(image_info['key']))
                                    cv2.imwrite(save_path, image_info["image"])
                                images = []
                        imageArray = {}
                        images = []
                else:
                    save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/wu", str(key))
                    cv2.imwrite(save_path, image)
                    up_imageArray = []
                    continuityArray = []
            count = count + 1
            logger.info("end ===============" + key)
            logger.info(str(count) + ",totail>>>" + str(time.time() - start))
            index += 1
       f_count = f_count + 1
    if 'index' in imageArray.keys():
        if imageArray['start'] < count - E_TOTAIL:
            save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/test", str(imageArray['key']))
            if imageArray['apm'] == False:
                if imageArray['count'] >= APM_TOTAIL:
                    imageArray['apm'] = True
                    cv2.imwrite(save_path, imageArray["image"])
                    base64_data = frame2base64(imageArray["image"])
                    data_json = {}
                    data_json["type"] = 2
                    data_json["image"] = "data:image/jpeg;base64," + base64_data.decode()
                    websocket_server.servers[0].send_message_to_all(str(json.dumps(data_json)))
                    ws_data_list.append(data_json)

                    for image_info in images:
                        save_path = "{}/{:>03s}.jpg".format("/home/woody/tmp/openpose/li", str(image_info['key']))
                        cv2.imwrite(save_path, image_info["image"])
                    images = []
            imageArray = {}
            images = []
    logger.info(">>>" + str(time.time() - start))
    video.release()
    logger.info("Totally save {:d} pics".format(index - 1))


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
    logger.info("calcHipAngle:" + str(realangle.real));
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
    logger.info("calcKneeAngle:" + str(realangle.real))
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
    logger.info("calcLenRate:" + str(result))
    if result >= 0.8 and result <= 1.2 :
        return result,False
    else:
        return result,True



def calcBuleRate(image, left0, left1, right0, right1):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvSplit = cv2.split(hsvImage)
    cv2.equalizeHist(hsvSplit[2], hsvSplit[2])
    cv2.merge(hsvSplit, hsvImage)
    thresholded = cv2.inRange(hsvImage, np.array(LOW_COLOR_SCALAR), np.array(HIGH_COLOR_SCALAR))

    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))

    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,element)

    thresholded = cv2.morphologyEx(thresholded,cv2.MORPH_CLOSE,element)

    rateValue = APM_BULE_SCORE

    box = np.array([[left0, left1, right0, right1]], dtype = np.int32)
    maskImage = np.zeros(image.shape[:2], dtype = "uint8")
    cv2.polylines(maskImage, box, 1, 255)
    totalArea = calcTotalArea(cv2.fillPoly(maskImage, box, 255))

    blueArea = calcBlueArea(cv2.bitwise_and(thresholded, thresholded, mask=maskImage))
    #cv2.imshow("maskImage",cv2.bitwise_and(thresholded, thresholded, mask=maskImage))
    #cv2.waitKey(0)
    mleft0 = [(left0[0]*2/3)+(left1[0]/3),(left0[1]*2/3)+(left1[1]/3)]
    mleft1 = [(left0[0]*1/3)+(left1[0]*2/3),(left0[1]*1/3)+(left1[1]*2/3)]
    mright1 = [(right0[0]*1/3)+(right1[0]*2/3),(right0[1]*1/3)+(right1[1]*2/3)]
    mright0 = [(right0[0]*2/3)+(right1[0]/3),(right0[1]*2/3)+(right1[1]/3)]
    mblueArea = 0
    mtotalArea = 0
    if totalArea == 0:
        return False, 0
    if (blueArea/totalArea) >= rateValue:
        return True, blueArea/totalArea

    if (blueArea/totalArea) < rateValue and rateValue == APM_BULE_SCORE:
        mbox = np.array([[mleft0,mleft1, mright0, mright1]], dtype = np.int32)
        mmaskImage = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.polylines(mmaskImage, mbox, 1, 255)
        mtotalArea = calcTotalArea(cv2.fillPoly(mmaskImage, mbox, 255))
        mblueArea = calcBlueArea(cv2.bitwise_and(thresholded, thresholded, mask=mmaskImage))

    #print(((blueArea-mblueArea)/(totalArea-mtotalArea)))
    if totalArea-mtotalArea == 0:
        return False, 0
    return ((blueArea-mblueArea)/(totalArea-mtotalArea)) >= rateValue, (blueArea-mblueArea)/(totalArea-mtotalArea)

def calcTotalArea(maskImage):
    _, binary = cv2.threshold(maskImage, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area = cv2.contourArea(contours[0])
    return area

def calcBlueArea(maskedImage):
    _, binary = cv2.threshold(maskedImage, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blueArea = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        blueArea += area
    return blueArea


def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    #server.send_message_to_all("Hey all, a new client has joined us")


# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))
    if message == "message":
        if ws_data_list is not None:
            for data_json in ws_data_list:
                server.send_message(client, str(json.dumps(data_json)))

def frame2base64(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #将每一帧转为Image
    output_buffer = BytesIO() #创建一个BytesIO
    img.save(output_buffer, format='JPEG') #写入output_buffer
    byte_data = output_buffer.getvalue() #在内存中读取
    base64_data = base64.b64encode(byte_data) #转为BASE64
    return base64_data #转码成功 返回base64编码



if __name__ == '__main__':
    #path = "/home/woody/tmp/openpose/fang/data800"
    pross()




