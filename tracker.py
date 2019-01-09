# coding: utf-8

'''
File: tracker.py
Project: AlphaPose
File Created: Thursday, 1st March 2018 6:12:23 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 1st October 2018 12:53:12 pm
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2018 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import cv2
from utils import *
import os.path
import copy
import numpy as np
class Tracker:


    def __init__(self, link=100, drop=2.0, num=7, mag=30, match=0.2, orb=1):
        # super parameters
        # 1. look-ahead LINK_LEN frames to find tracked human bbox
        # 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
        # 3. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score(Non DeepMatching)
        # 4. drop low-score(<DROP) keypoints
        # 5. pick high-score(top NUM) keypoints when computing pose_IOU
        # 6. box width/height around keypoint for computing pose IoU
        # 7. match threshold in Hungarian Matching
        # 9. use orb matching or not
        self.lastframe = None
        self.link_len = link
        self.weights = [1,2,1,2,0,0]
        self.weights_fff = [0,1,0,1,0,0]
        self.drop = drop
        self.num = num
        self.mag = mag
        self.match_thres = match
        self.use_orb = orb
        self.max_pid_id = 0
        self.total_track = {}

    def track(self, frame, notrack):

        track = {}

        #  json file without tracking information
        # Note: time is a little long, so it is better to uncomment the following save operation at first time

        for imgpath in sorted(notrack.keys()):
            vname, fname = os.path.split(imgpath)
            track[vname] = {}
            if not vname in self.total_track:
                self.total_track[vname]= {}

            track[vname][fname] = {'num_boxes':len(notrack[imgpath])}
            for bid in range(len(notrack[imgpath])):

                track[vname][fname][bid+1] = {}
                track[vname][fname][bid+1]['box_score'] = notrack[imgpath][bid]['score']
                track[vname][fname][bid+1]['box_pos'] = self.get_box(notrack[imgpath][bid]['keypoints'], frame)
                track[vname][fname][bid+1]['box_pose_pos'] = np.array(notrack[imgpath][bid]['keypoints']).reshape(-1,3)[:,0:2]
                track[vname][fname][bid+1]['box_pose_score'] = np.array(notrack[imgpath][bid]['keypoints']).reshape(-1,3)[:,-1]
            self.total_track[vname][fname] = {}
            self.total_track[vname][fname] = track[vname][fname]

        track_single, track = track, self.total_track
        if self.lastframe is None:
            self.lastframe = frame
            return None, None

        # tracking process

        for video_name in track_single.keys():
            frame_list = sorted(list(track[video_name].keys()))
            t_idx = len(frame_list) - 2
            frame_name = frame_list[t_idx]
            for idx, frame_name in [[t_idx, frame_name]]:
                next_frame_name = frame_list[idx+1]
                # init tracking info of the first frame in one video
                if idx == 0:
                    for pid in range(1, track[video_name][frame_name]['num_boxes']+1):
                        track[video_name][frame_name][pid]['new_pid'] = pid
                        track[video_name][frame_name][pid]['match_score'] = 0

                max_pid_id = max(self.max_pid_id, track[video_name][frame_name]['num_boxes'])

                img1 = self.lastframe
                img2 = frame
                self.lastframe = frame

                ret = []
                # Initiate ORB detector
                #todo neatures=10000
                orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)

                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)

                if len(kp1) * len(kp2) < 400:
                    height, width, channels = img1.shape
                    for x in range(width):
                        for y in range(height):
                            ret.append([x, y, x, y, 1.0])
                    return np.array(ret)

                # FLANN parameters
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=12,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=2)  # 2
                #todo checks=100
                search_params = dict(checks=100)  # or pass empty dictionary

                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(des1, des2, k=2)

                # ratio test as per Lowe's paper
                for i, m_n in enumerate(matches):
                    if len(m_n) != 2:
                        continue
                    elif m_n[0].distance < 0.80 * m_n[1].distance:
                        ret.append([kp1[m_n[0].queryIdx].pt[0], kp1[m_n[0].queryIdx].pt[0], kp2[m_n[0].trainIdx].pt[0],
                                    kp2[m_n[0].trainIdx].pt[1], m_n[0].distance])

                if len(ret) < 50:
                    height, width, channels = img1.shape
                    ret = []
                    for x in range(width):
                        for y in range(height):
                            ret.append([x, y, x, y, 1.0])

                all_cors = np.array(ret)

                # if there is no people in this frame, then copy the info from former frame
                if track[video_name][next_frame_name]['num_boxes'] == 0:
                    track[video_name][next_frame_name] = copy.deepcopy(track[video_name][frame_name])
                    continue

                cur_all_pids, cur_all_pids_fff = stack_all_pids(track[video_name], frame_list[:-1], idx, max_pid_id, self.link_len)
                match_indexes, match_scores = best_matching_hungarian(
                    all_cors, cur_all_pids, cur_all_pids_fff, track[video_name][next_frame_name], self.weights, self.weights_fff, self.num, self.mag)

                for pid1, pid2 in match_indexes:
                    if match_scores[pid1][pid2] > self.match_thres:
                        track[video_name][next_frame_name][pid2+1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                        max_pid_id = max(max_pid_id, track[video_name][next_frame_name][pid2+1]['new_pid'])
                        track[video_name][next_frame_name][pid2+1]['match_score'] = match_scores[pid1][pid2]

                # add the untracked new person
                for next_pid in range(1, track[video_name][next_frame_name]['num_boxes'] + 1):
                    if 'new_pid' not in track[video_name][next_frame_name][next_pid]:
                        max_pid_id += 1
                        track[video_name][next_frame_name][next_pid]['new_pid'] = max_pid_id
                        track[video_name][next_frame_name][next_pid]['match_score'] = 0
                self.max_pid_id = max_pid_id

                if (len(frame_list)) > self.link_len:
                    del track[video_name][frame_list[0]]
        # return track
        return copy.deepcopy(track[video_name][frame_name]), img1

    def get_box(self, pose, img):
        img_height, img_width, _ = img.shape
        pose = np.array(pose).reshape(-1, 3)
        x = pose[:, 0]
        y = pose[:, 1]
        xmin = np.min(x[x > 0])
        xmax = np.max(pose[:, 0])
        ymin = np.min(y[y > 0])
        ymax = np.max(pose[:, 1])

        return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)

    def wrapped_track(self, frame, keypoints, scores, video_name, ordinal):
        notrack = {}
        imgpath = "%s/%08d.jpg" % (video_name, ordinal)
        notrack[imgpath] = []
        for i in range(len(keypoints)):
            notrack[imgpath].append({"keypoints": keypoints[i], "score": scores[i]})

        # track, last_frame = self.track(frame, notrack)
        # if track is not None:
        #     frame_info = copy.deepcopy(track[video_name][list(track[video_name].keys())[0]])
        #     del frame_info["num_boxes"]
        frame_info, last_frame = self.track(frame, notrack)
        if frame_info is not None:
            del frame_info["num_boxes"]
            return sorted(list(frame_info.values()), key=lambda x: x["new_pid"]), last_frame
        else:
            return None,None
