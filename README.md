# pose_estimation

This repository is inspired by "" paper. 

## Code directories

mpii - Directory for mpii images and annotations.

python - This directory contains python scripts for preprocessing mpii dataset images.

c++ - C++ code for training and prediction using SVM.

jsons - Directory to keep intermediate json files

pickles - Directory to keep intermediate pickle files

experimental_notebooks - Jupyter notebooks used for data exploration and experimentation

Dataset

Download the MPII human pose dataset from http://human-pose.mpi-inf.mpg.de/#download and put the extracted images in mpii/images folder.

## Preprocessing

For preprocessing images go to python directory and run preprocessing.py script

$ cd python 
$ ./preprocessing


### Preprocessing steps

1. Convert mpii mat file into json file
2. Select images with right hand joint visibles. This step selects around 12K images.
3. Crop and resize image into 96x96 block with center as right elbow.
4. Group images into cluster based on inter cluster joint distance. Only use cluster with more than 100 images for training. This step creates 28 pose clusters. This steps also selects negative images for each cluster.
5. Calculate joint position for 96x96 cropped image. 


## SVM training

For training SVM go to c++ directory, build and run TrainSVM 

$ cd c++
$ cmake . 
$ make 
$ ./TrainSVM

### SVM training steps 

1. For each cluster read positive and negative images from mpii/cropped_imgs directory.
2. Convert the image to gray scale and calculate HOG features.
3. Train dlib classifier using HOG features. 
4. Serialize trained SVM and put in trained_svms directory


## SVM testing

For testing SVM, go to c++ directory and run TestSVM (assuming TrainSVM is executed)

$ cd c++
$ ./TestSVM

### SVM testing steps

In current implementation test image is harcoded in c++ code. 

1. Read test image from mpii/images directory.
2. Extract the bounding box using location of head and hips.
3. Expand the bounding box region by 3 times.
4. Calculate sliding windows for bounding box.
5. Calculate HOG features for each sliding window and run SVM classifiers.
6. Select the best cluster/sliding window.
7. Print the centroid of the best cluser.
8. Mark right hand joints using hte cluster centroid in the original test image. 


## TODO

* Use image patch and poselets as features. 
* Use PCP metric for classification score
* Instead of using all the three joint for right hand divide it into upper and lower arms. 
* Repeat the process for left hand
* Improve test code to take input as image and bounding box dimensions. 