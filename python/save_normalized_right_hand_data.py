#!/usr/bin/env python

import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import pickle
from copy import deepcopy

from util import calculate_center, calculate_sigma


# This scripts selects images from mpii dataset that have visible
# right hand joints


def display_img(img):
    img = img[...,::-1]
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.show()
    
def draw_pts(img, x, y):
    #print(x, " ", y)
    img = cv2.circle(img, (int(x), int(y)), 2, (255,0,0), 3)
    return img

def count_right_side_visible(data_file):
    count = 0
    for line in data_file:
        line_data = json.loads(line)

        img = cv2.imread(img_dir+line_data['filename'])
        is_visible = line_data['is_visible']
        if is_visible['r_wrist'] == 1 and is_visible['r_elbow'] == 1 and is_visible['r_shoulder'] ==1:
            count = count + 1
    return count


def normalize_joints(joint_pos):
    center_x, center_y = calculate_center(joint_pos)
    sigma = calculate_sigma(joint_pos)
        
    for i in joint_pos.keys():
        joint_pos[i][0] = (joint_pos[i][0] - center_x)/sigma
        joint_pos[i][1] = (joint_pos[i][1] - center_y)/sigma
        
    return joint_pos

def grp_right_side_visible(data_file):
    img_dir = "mp2/images/"
    count = 0
    img_lst = []
    joint_lst = []
    orig_joint_lst = []
    for line in data_file:
        #if count == 5:
        #    break
        count += 1
        line_data = json.loads(line)
        filename = line_data['filename']
        print("Processing "+ filename)
        img = cv2.imread(img_dir+line_data['filename'])
        is_visible = line_data['is_visible']
        if is_visible['r_wrist'] == 1 and is_visible['r_elbow'] == 1 and is_visible['r_shoulder'] == 1:
            img_lst.append(filename)
            joint_arr = []
            orig_joint_pos = line_data['joint_pos']
            orig_joint_lst.append(deepcopy(orig_joint_pos))
            
            joint_pos = normalize_joints(deepcopy(orig_joint_pos))
            #joint_pos = orig_joint_pos
            joint_arr.append(joint_pos['r_elbow'])
            joint_arr.append(joint_pos['r_shoulder'])
            joint_arr.append(joint_pos['r_wrist'])
            joint_arr.append(joint_pos['l_shoulder'])
            joint_lst.append(joint_arr)
    #print(joint_lst)
    #print(orig_joint_lst)
    return img_lst, joint_lst, orig_joint_lst

def convert_img_data():
    img_data = pickle.load(open('../pickles/img_data.p', 'rb'))
    json_str = json.dumps(img_data)
    open('../jsons/img_data.json', 'w').write(json_str)


def save_normalized_right_hand_data():
    data = open("../mpii/annotation/data.json")
    img_dir = "../mpii/images/"
    img_lst, joint_lst, orig_joint_lst = grp_right_side_visible(data)

    print(len(img_lst))
    print(joint_lst[0])
    print(len(joint_lst))
    res = {"img_lst": img_lst, "joint_lst": joint_lst, "orig_joint_lst": orig_joint_lst}

    pickle.dump( res, open( "../pickles/img_data.p", "wb" ) )

if __name__ == "__main__":
    save_normalized_right_hand_data()