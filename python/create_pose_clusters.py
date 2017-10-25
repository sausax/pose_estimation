#!/usr/bin/env python

import scipy.io as sio
import scipy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering


## This script create cluster for right hand visible images based on 
## closeness of joints among cluster elements

def create_cluster(keypoints):
    centers = {}
    points_map = {}
    centers[0] = keypoints[0]
    points_map[0] = []
    for indx, keypoint in enumerate(keypoints):
        point_added = False
        min_score = 0.8
        choosen_center = -1
        for center_indx, center_keypoint in centers.items():
            dist = np.linalg.norm(center_keypoint-keypoint)
            if dist < min_score:
                point_added = True
                min_score = dist
                choosen_center = center_indx
                
        if point_added == False:
            centers[indx] = keypoint
            points_map[indx] = [indx]
        else:
            points_map[choosen_center].append(indx)

    return points_map

def count_significant_grps(points_map, significant_size):
    total_size = 0
    count = 0
    grps = []
    for key, val in points_map.items():
        if len(val) >= significant_size:
            count += 1
            total_size += len(val)
            grps.append(val)
    return grps, count, total_size

def realign_center(point_grps, keypoints):
    points_map = {}
    # Iterating over point groups
    # to find best center for each group
    for points in point_grps:
        min_dist = 100000
        max_point = points[0] # Point which is farthest away from center for this grop
        curr_center = points[0]
        for center_point in points:
            total_dist = 0
            max_dist = 0
            max_dist_point = center_point # Point in the group which is farthest from the current center
            for point in points:
                curr_dist = np.linalg.norm(keypoints[center_point]-keypoints[point])
                if curr_dist > max_dist:
                    max_dist = curr_dist
                    max_dist_point = point
                total_dist += curr_dist
                
            if total_dist < min_dist:
                curr_center = center_point
                min_dist = total_dist
                max_point = max_dist_point
        points_map[curr_center] = (points, max_point)
    return points_map


# Add negative examples for each center
# Iterate over point map
# For each center, iterate over other centers
# if the dist of a point in other point group is more than
# 2 times the max dist for inner grp than add that point in the
# list of negative examples
def prepare_training_data(points_map, keypoints):
    training_data_map = {}
    for center_point in points_map.keys():
        points, max_point = points_map[center_point]
        total_pos_points = len(points)
        max_dist = 2*np.linalg.norm(keypoints[center_point]-keypoints[max_point]) 
        neg_points = []
        neg_points_full = False
        for neg_center_point in points_map.keys():
            if center_point == neg_center_point:
                continue
            
            curr_neg_points, _ = points_map[neg_center_point]
            for neg_point in curr_neg_points:
                neg_dist = np.linalg.norm(keypoints[center_point]-keypoints[neg_point])
                if neg_dist > max_dist:
                    neg_points.append(neg_point)
                    
                if len(neg_points) >= total_pos_points:
                    neg_points_full = True
                    break
                    
            if neg_points_full:
                break
                
        training_data_map[center_point] = (points, neg_points)
        
    return training_data_map

def convert_training_data(trainint_data):
    json_obj = {}

    for key in training_data:
        (pos_lst, neg_lst) = training_data[key]
        json_obj[key] = {}
        json_obj[key]['pos'] = pos_lst
        json_obj[key]['neg'] = neg_lst

    json_str = json.dumps(json_obj)
    open('../jsons/training_data.json', 'w').write(json_str)
            

def create_pose_clusters():
    img_data = pickle.load( open( "../pickles/img_data.p", "rb" ) )


    joint_lst = img_data['joint_lst']
    joint_arr = np.array(joint_lst)

    joint_arr_reshaped = joint_arr.reshape(-1, 8)

    img_lst = img_data['img_lst']
    orig_joint_lst = img_data['orig_joint_lst']
                    
    points_map = create_cluster(joint_arr_reshaped)

    grps_with_100_elems, _, _ = count_significant_grps(points_map, 100)
    realigned_points = realign_center(grps_with_100_elems, joint_arr_reshaped)

    print(realigned_points.keys())

    training_data_map = prepare_training_data(realigned_points, joint_arr_reshaped)
    pickle.dump(training_data_map, open( "../pickles/training_data.p", "wb" ) )
    convert_training_data(training_data_map)


if __name__ == "__main__":
    create_pose_clusters()