#import init as init #has file paths
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2
import numpy as np
import functions.helper_modules as functions
import random


def vis_boxes_and_move(scene_path):
    """ Visualizes bounding boxes and images in the scene.

    Allows user to navigate the scene via the movement 
    pointers using the keyboard


    ARGUMENTS:
        scene_path: the string full path of the scene to view
            Ex) vis_camera_pos_dirs('/path/to/data/Home_01_1')

    """


    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    annotations_path = os.path.join(scene_path,'annotations.json')
    depth_path = os.path.join(scene_path, 'high_res_depth')

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)
    depth_names = os.listdir(images_path)
    depth_names.sort()


    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = ''
    
    #image parameters
    reach = False
    bump = 0
    min_distance = 500
    obstacle_side = []      #obstacle_side[0] = left partition
                            #obstacle_side[1] = middle partition
                            #obstacle_side[2] = right partition
    
    
 

    while not reach:
        cur_depth_name = functions.name_RGB2DEPTH(cur_image_name)
        #load the current image and annotations 
        rgb_image = cv2.imread(os.path.join(images_path,cur_image_name))
        depth_image =  cv2.imread(os.path.join(depth_path, cur_depth_name), cv2.IMREAD_ANYDEPTH)
        

        #plot the image and draw the boxes
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 960, 540)
        cv2.imshow('image', rgb_image)


        obstacle_side = functions.obs_detect(depth_image, min_distance)
        print(obstacle_side)
        key = cv2.waitKey(300)
        


        
        '''
        #if all partition has lower threshold
        if obstacle_side[0] == 1 and obstacle_side[1] == 1 and obstacle_side[2] == 1:
            #rotate 180 degree
            next_image_name = annotations[cur_image_name]['rotate_cw']
            next_image_name = annotations[next_image_name]['rotate_cw']
            next_image_name = annotations[next_image_name]['rotate_cw']
            next_image_name = annotations[next_image_name]['rotate_cw']
            next_image_name = annotations[next_image_name]['rotate_cw']
            next_image_name = annotations[next_image_name]['rotate_cw']
        '''
 
        
        #else if left partition has lower threshold
        if obstacle_side[0]:
            #shifts towards the right side
                next_image_name = annotations[cur_image_name]['backward']
                if next_image_name == '':
                    next_image_name = cur_image_name
                    next_image_name = annotations[cur_image_name]['rotate_cw']
                    next_image_name = annotations[next_image_name]['rotate_cw']
                elif next_image_name != '':
                    next_image_name = annotations[cur_image_name]['rotate_cw']
                    next_image_name = annotations[next_image_name]['rotate_cw']
            
        #else if right partition has lower threshold
        elif obstacle_side[2]:
            #shifts towards the right side
                next_image_name = annotations[cur_image_name]['backward']
                if next_image_name == '':
                    next_image_name = cur_image_name
                    next_image_name = annotations[cur_image_name]['rotate_ccw']
                    next_image_name = annotations[next_image_name]['rotate_ccw']
                elif next_image_name != '':
                    next_image_name = annotations[cur_image_name]['rotate_ccw']
                    next_image_name = annotations[next_image_name]['rotate_ccw']
            
        #else if middle partition has lower threshold
        elif obstacle_side[1]:
            #shift towards the left or the right side by random
            choose_dir = random.randint(0,1)
            if choose_dir == 0:
                next_image_name = annotations[cur_image_name]['backward']
                next_image_name = annotations[next_image_name]['left']
            elif choose_dir == 1:
                next_image_name = annotations[cur_image_name]['backward']
                next_image_name = annotations[next_image_name]['right']
        
                
        else:
            
            next_image_name = annotations[cur_image_name]['forward']
            

        #If there is an image available, continue navigating forward
        if next_image_name != '':
            cur_image_name = next_image_name
            
        elif next_image_name == '':
            #shift towards the left or the right side by random
            choose_dir = random.randint(0,1)
            if choose_dir == 0:
                cur_image_name = annotations[cur_image_name]['rotate_ccw']
                cur_image_name = annotations[cur_image_name]['rotate_ccw']
                cur_image_name = annotations[cur_image_name]['rotate_ccw']
                cur_image_name = annotations[cur_image_name]['rotate_ccw']
            elif choose_dir == 1:
                cur_image_name = annotations[cur_image_name]['rotate_cw']
                cur_image_name = annotations[cur_image_name]['rotate_cw']
                cur_image_name = annotations[cur_image_name]['rotate_cw']
                cur_image_name = annotations[cur_image_name]['rotate_cw']
            
            
            
        if key == 113:
            break
            cv2.destroyAllWindows()
    



