import scipy.io as sio
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
import pickle
from copy import deepcopy

def calculate_center(joint_pos):
    center_x = 0
    center_y = 0
    for val in joint_pos.values():
        center_x += val[0]
        center_y += val[1]
        
    center_x = center_x/len(joint_pos)
    center_y = center_y/len(joint_pos)
    
    return center_x, center_y

# Calculate sigma for image keypoints
def calculate_sigma(joint_pos):
    sigma = 0
    center_x, center_y = calculate_center(joint_pos)
    for i in ['r_elbow', 'r_shoulder', 'r_wrist', 'l_shoulder']:
        x_diff = joint_pos[i][0] - center_x
        y_diff = joint_pos[i][1] - center_y
        sigma = sigma + x_diff*x_diff + y_diff*y_diff
    
    sigma = 0.25 * sigma
    sigma = math.sqrt(sigma)
    return sigma

def display_img(img):
    img = img[...,::-1]
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.show()
    
def fix_size(max_val, curr_val):
    if curr_val < 0:
        return 0
    elif curr_val > max_val:
        return max_val
    else:
        return curr_val

