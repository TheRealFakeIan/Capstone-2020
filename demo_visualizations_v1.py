# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:58:59 2020

@author: cyeeh
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:57:14 2020

@author: cyeeh
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2
import math
import open3d as o3d
import numpy as np
#import pcl


#intrinsic parameters (infrared kinect v2 approx )
#for depth
# =============================================================================
# fx = 366.193  # focal length x
# fy = 366.193  # focal length y
# cx = 256.684 # optical center x
# cy = 207.085  # optical center y
# scale = 1000 # scaling parameter, 1mm = 0.001m
# =============================================================================
#for RGB
# =============================================================================
# fx = 1081.37  # focal length x
# fy = 1081.37 # focal length y
# cx = 959.5 # optical center x
# cy = 539.5  # optical center y
# scale = 1 # scaling parameter, 1mm = 0.001m
# camera_intrinsic= np.array([[fx, 0, cx], [0, fx, cy],[0, 0, 1]])
# =============================================================================


# =============================================================================
# rgb = cv2.imread("rgb.png")
# depth = cv2.imread( "depth.png", -1 )
# 
# 
# 
# 
# 
# plt.subplot(1, 2, 1)
# plt.title('grayscale image')
# plt.imshow(target_rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title(' depth image')
# plt.imshow(target_rgbd_image.depth)
# plt.show()
# =============================================================================


def vis_draw_trajectory(scene_path):
    

    def get_corresponding_depth(image_name):
        return image_name[:-5]+'3'+image_name[-4:]
    
    #intrinsic parameters (infrared kinect v2 approx )
    fx = 1081.37  # focal length x
    fy = 1081.37 # focal length y
    cx = 959.5 # optical center x
    cy = 539.5  # optical center y
    scale = 1 # scaling parameter, 1mm = 0.001m
    camera_intrinsic= np.array([[fx, 0, cx], [0, fx, cy],[0, 0, 1]])

    ffd = cv2.FastFeatureDetector_create(threshold=25,nonmaxSuppression=True)      
    lk_params = dict( winSize  = (50,50),
          maxLevel = 30,
          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # mention the Optical Flow Algorithm parameters    
        

    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    depth_path = os.path.join(scene_path,'high_res_depth')
    annotations_path = os.path.join(scene_path,'annotations.json')
    

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    depth_names = os.listdir(depth_path)
    depth_names.sort()

    ann_file = open(annotations_path)
    annotations = json.load(ann_file)


    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = ''
    tminus1_color_image = None
    
    #set up position and rotation matrix
    current_pos = np.zeros((3, 1))
    current_rot = np.eye(3)

    # create graph.
    position_figure = plt.figure()
    position_axes = position_figure.add_subplot(1, 1, 1)        
    position_axes.set_aspect('equal', adjustable='box')
    flag_same_image = 0
    while True:

        #Read images 
        t_color_image = cv2.imread(os.path.join(images_path,cur_image_name))
        t_gray_image = cv2.cvtColor(t_color_image, cv2.COLOR_BGR2GRAY) #BGR -> Grayscale
        
        #Detect features from RGB image using FAST algorithm
                                                     
        #t_keypoints = ffd.detect(t_color_image,None)
        t_keypoints = ffd.detect(t_gray_image,None)

        #first image of our image sequence
        if tminus1_color_image is None:
            tminus1_color_image = t_color_image
            tminus1_keypoints = t_keypoints
            tminus1_gray_image = t_gray_image
            
            
        points = np.array([k.pt for k in tminus1_keypoints],dtype=np.float32)

        if flag_same_image != 1:
        
            #track detected features in the current gray image using Lucas Kanade Optical Flow algorithm
            p1, st, err = cv2.calcOpticalFlowPyrLK(tminus1_gray_image, t_gray_image, 
                                                            points, None, **lk_params)
            
            #detect and track again if threshold not cleared
            no_of_features = 100
            while(len(p1) < no_of_features):
                tminus1_keypoints = ffd.detect(tminus1_gray_image,None)
                p1, st, err = cv2.calcOpticalFlowPyrLK(tminus1_gray_image, t_gray_image, 
                                                            points, None, **lk_params)
            
            E, mask = cv2.findEssentialMat(points, p1, camera_intrinsic,
                                           cv2.RANSAC, 0.999, 1.0, None)
            
            points, R, t, mask = cv2.recoverPose(E, p1, points, camera_intrinsic)
    
            
            current_pos += current_rot.dot(t) / scale
            current_rot = R.dot(current_rot)
    
            position_axes.scatter(current_pos[0][0], current_pos[2][0])
            plt.pause(.01)
            position_figure.savefig("position_plot.png")
            
        img = cv2.drawKeypoints(t_gray_image, t_keypoints, None)
            
            
        cv2.namedWindow('bgr image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('bgr image', 960, 540) #original image resolution is 1080x1920(h x w)
        cv2.imshow('bgr image',img)
        
        
        
# =============================================================================
#         #no. of rows, columns, and channels
#         h,w,d = bgr_image.shape 
#         cv2.namedWindow('bgr image', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('bgr image', 960, 540) #original image resolution is 1080x1920(h x w)
#         cv2.imshow('bgr image',bgr_image)
# 
#         traj_matrix = np.zeros((h,w,d))
# =============================================================================

# =============================================================================
#         #window for combined rgbd image
#         cv2.namedWindow('normalised rgbd depth', cv2.WINDOW_AUTOSIZE) #shown with original size
#         #cv2.resizeWindow('depth image', 960, 540)
#         cv2.imshow('depth image',depth_image)
#         
#         #window for drawing trajectory
#         cv2.namedWindow('trajectory', cv2.WINDOW_AUTOSIZE)
# =============================================================================

            
        
        
        key = cv2.waitKey(-1)
    
        if key==119:
            next_image_name = annotations[cur_image_name]['forward']
        elif key==97:
            next_image_name = annotations[cur_image_name]['rotate_ccw']
        elif key==115:
            next_image_name = annotations[cur_image_name]['backward']
        elif key==100:
            next_image_name = annotations[cur_image_name]['rotate_cw']
        elif key==101:
            next_image_name = annotations[cur_image_name]['left']
        elif key==114:
            next_image_name = annotations[cur_image_name]['right']
        elif key==104:
            next_image_name = cur_image_name
            print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                  "Enter a character to move around the scene:",
                  "'w' - forward", 
                  "'a' - rotate counter clockwise", 
                  "'s' - backward", 
                  "'d' - rotate clockwise", 
                  "'e' - left", 
                  "'r' - right", 
                  "'q' - quit", 
                  "'h' - print this help menu"))
        elif key==113:
            cv2.destroyAllWindows
            break



        #if the user inputted move is valid (there is an image there) 
        #then update the image to display. If the move was not valid, 
        #the current image will be displayed again
            
            
        if next_image_name != '':
            tminus1_color_image = t_color_image
            tminus1_gray_image = t_gray_image
            tminus1_keypoints = t_keypoints
            cur_image_name = next_image_name
            flag_same_image = 0
        
        else:
            flag_same_image = 1
            print("Dead end. Please turn around. ")
            
            

    
    
def vis_navigate(scene_path):

    def obs_detect(depth_image, threshold = 500):
        left_flag = 0
        mid_flag = 0
        right_flag = 0    
        copy = depth_image.copy()
        second_smallest = sorted(list(set(copy.flatten().tolist())))[2]
        result = np.where(t_depth_image == second_smallest)
        coord_list = list(zip(result[0],result[1]))
        for row,col in coord_list:
            if depth_image[row][col] < threshold:
                if col < 640: 
                    left_flag = 1
                elif col < 1280: 
                    mid_flag = 1
                else:
                    right_flag = 1
        if left_flag:
            print("obstacle detected on the left")
        if right_flag:
            print("obstacle detected on the right")
        if mid_flag:
            print("obstacle detected in the middle")
        return [left_flag,mid_flag,right_flag]
    
    def get_corresponding_depth(image_name):
        return image_name[:-5]+'3'+'.png'
    
    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    depth_path = os.path.join(scene_path,'high_res_depth')
    annotations_path = os.path.join(scene_path,'annotations.json')
    

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    depth_names = os.listdir(depth_path)
    depth_names.sort()

    ann_file = open(annotations_path)
    annotations = json.load(ann_file)


    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = ''
 
    
    while True:
        
        cur_depth_name = get_corresponding_depth(cur_image_name)
        #read images
        t_depth_image = cv2.imread(os.path.join(depth_path,cur_depth_name), cv2.IMREAD_ANYDEPTH)
        t_color_image = cv2.imread(os.path.join(images_path,cur_image_name))
        
        obs_detect(t_depth_image)
        
        cv2.namedWindow('bgr image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('bgr image', 960, 540) #original image resolution is 1080x1920(h x w)
        cv2.imshow('bgr image',t_color_image)
        
        #cv2.namedWindow('depth image', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('depth image', 960, 540) #original image resolution is 1080x1920(h x w)
        #cv2.imshow('depth image',t_depth_image)

        key = cv2.waitKey(-1)
    
        if key==119:
            next_image_name = annotations[cur_image_name]['forward']
        elif key==97:
            next_image_name = annotations[cur_image_name]['rotate_ccw']
        elif key==115:
            next_image_name = annotations[cur_image_name]['backward']
        elif key==100:
            next_image_name = annotations[cur_image_name]['rotate_cw']
        elif key==101:
            next_image_name = annotations[cur_image_name]['left']
        elif key==114:
            next_image_name = annotations[cur_image_name]['right']
        elif key==104:
            next_image_name = cur_image_name
            print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(
                  "Enter a character to move around the scene:",
                  "'w' - forward", 
                  "'a' - rotate counter clockwise", 
                  "'s' - backward", 
                  "'d' - rotate clockwise", 
                  "'e' - left", 
                  "'r' - right", 
                  "'q' - quit", 
                  "'h' - print this help menu"))
        elif key==113:
            cv2.destroyAllWindows
            break
        
        if next_image_name != '':
            cur_image_name = next_image_name
            
    
    return
    
