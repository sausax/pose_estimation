#!/usr/bin/env python

from mpii_dataset import save_joints
from save_normalized_right_hand_data import save_normalized_right_hand_data
from save_cropped_images import save_cropped_images
from create_pose_clusters import create_pose_clusters
from update_joint_loc_for_new_origin import update_joint_location


print("Transforming mpii mat file to json")
save_joints()

print("Saving images data that have right hand joints marked visible")
save_normalized_right_hand_data()

print("Cropping images with right elbow as center")
save_cropped_images()

print("Creating cluster of poses based on inter-cluster joint distance")
create_pose_clusters()

print("Updating joint location in cropped 96x96 cropped image")
update_joint_location()