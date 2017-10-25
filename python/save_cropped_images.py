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

from util import calculate_center, calculate_sigma, display_img

# This script crops the visible right hand joint images into 96x96 dimension
# This is a preprocessing step before training classifier over the images

# Extract image around center by sigma
def show_subsection(img, sigma, center_x, center_y):
    x_start = int(center_x-sigma)
    x_end = int(center_x+sigma)
    y_start = int(center_y-sigma)
    y_end = int(center_y+sigma)
    crop_img = img[y_start:y_end, x_start:x_end]
    display_img(crop_img)
    img = cv2.rectangle(img,(x_start, y_start),(x_end,y_end),(0,255,0),3)
    display_img(img)
    
def draw_bounding_box(img_file, joints):
    img_dir = "../mpii/images/"
    img = cv2.imread(img_dir+ img_file)
    center_x, center_y = calculate_center(joints)
    sigma = calculate_sigma(joints)
    show_subsection(img, sigma, joints['r_elbow'][0], joints['r_elbow'][1])
    
def fix_size(max_val, curr_val):
    if curr_val < 0:
        return 0
    elif curr_val > max_val:
        return max_val
    else:
        return curr_val
    
def crop_and_save(indx, img_file, joints):
    img_dir = "../mpii/images/"
    cropped_img_dir = "../mpii/cropped_imgs/"
    img = cv2.imread(img_dir+ img_file)
    center_x, center_y = (joints['r_elbow'][0], joints['r_elbow'][1])
    sigma = calculate_sigma(joints)
    
    max_x = img.shape[1]
    max_y = img.shape[0]
    
    x_start = fix_size(max_x, int(center_x-sigma))
    x_end = fix_size(max_x, int(center_x+sigma))
    y_start = fix_size(max_y, int(center_y-sigma))
    y_end = fix_size(max_y, int(center_y+sigma))
    #print(x_start, x_end, y_start, y_end, sigma)
    cropped_img = img[y_start:y_end, x_start:x_end]
    cropped_img = cv2.resize(cropped_img, (96, 96))
    cv2.imwrite(cropped_img_dir+str(indx)+'.jpeg',cropped_img)


def save_cropped_images():
    img_data = pickle.load( open( "../pickles/img_data.p", "rb" ) )
    for i in range(0, len(img_data['img_lst'])):
        if i % 100 == 0:
            print('Processed %d images' %(i))
        crop_and_save(i, img_data['img_lst'][i], img_data['orig_joint_lst'][i])


if __name__ == "__main__":
    save_cropped_images()