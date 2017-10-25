#!/usr/bin/env python

import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
from copy import deepcopy
import pickle

from util import calculate_center, calculate_sigma, fix_size


# This script calculates joint position in the cropped 96x96 images 

def update_joint_pos_using_center(img_file, joint_pos, center_x, center_y, sigma):
    img_center_x = joint_pos['r_elbow'][0]
    img_center_y = joint_pos['r_elbow'][1]
    img_dir = "../mpii/images/"
    img = cv2.imread(img_dir+img_file)
    max_x = img.shape[1]
    max_y = img.shape[0]
    x_start = fix_size(max_x, (img_center_x - sigma))
    x_end = fix_size(max_x, (img_center_x+sigma))
    y_start = fix_size(max_y, (img_center_y-sigma))
    y_end = fix_size(max_y, (img_center_y+sigma))
    
    height = y_end - y_start
    width = x_end - x_start
    
    if width == 0:
        print('max_x: ', max_x, 'center_x: ', center_x, 'sigma: ', sigma)
    
    new_joint_loc = {}
    for joint in ['r_elbow', 'r_wrist', 'r_shoulder', 'l_shoulder']:
        new_x = joint_pos[joint][0] - x_start
        new_x = (new_x/width)*96
        
        new_y = joint_pos[joint][1] - y_start
        new_y = (new_y/height)*96
        
        new_joint_loc[joint] = [new_x, new_y]
        
    return new_joint_loc

def update_keypoint_locations(img_lst, joint_lst):
    new_joint_lst = []
    # Iterate over right hand saved images
    for indx, joint_dict in enumerate(joint_lst):
        print('Processing index: ', indx)
        # Calculate the center using keypoints
        center_x, center_y = calculate_center(joint_dict)                       
        sigma = calculate_sigma(joint_dict)
        new_joint_map = update_joint_pos_using_center(img_lst[indx], joint_dict, center_x, center_y, sigma)
        new_joint_lst.append(new_joint_map)
        
    return new_joint_lst

def convert_new_joint_lst():
    new_joint_lst = pickle.load(open('new_joint_lst.p', 'rb'))
    json_str = json.dumps(new_joint_lst)
    open('jsons/new_joint_lst.json', 'w').write(json_str)
        
def update_joint_location():
    img_data = pickle.load(open("../pickles/img_data.p", "rb"))
    new_joint_lst = update_keypoint_locations(img_data['img_lst'], img_data['orig_joint_lst'])
    pickle.dump(new_joint_lst, open('../pickles/new_joint_lst.p', 'wb'))
    convert_new_joint_lst()

if __name__ == "__main__":
    update_joint_location()