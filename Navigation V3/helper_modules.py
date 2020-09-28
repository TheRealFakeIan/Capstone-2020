# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:27:34 2020

@author: user
"""
import os
import numpy as np


def check_present_instance(scene_path, requested_instance):
    present_instance_path = os.path.join(scene_path, 'present_instance_names.txt')
    present_instances = open(present_instance_path, 'r')
    present_instances = present_instances.readlines()
    
    present_instances_list = []
    for line in present_instances:
        present_instances_list.append(line.strip())
        
    present=False;
    for instance in present_instances_list:
        if instance == requested_instance:
            present=True;
            break
    return present


def command(move_command, cur_image_name, annotations):
    #get the next image name to display based on the 
    #user input, and the annotation.
    if move_command == 'w':
        next_image_name = annotations[cur_image_name]['forward']
    elif move_command == 'a':
        next_image_name = annotations[cur_image_name]['rotate_ccw']
    elif move_command == 's':
        next_image_name = annotations[cur_image_name]['backward']
    elif move_command == 'd':
        next_image_name = annotations[cur_image_name]['rotate_cw']
    elif move_command == 'e':
        next_image_name = annotations[cur_image_name]['left']
    elif move_command == 'r':
        next_image_name = annotations[cur_image_name]['right']

        
    return next_image_name

def name_RGB2DEPTH(image_name):
    return image_name[:-5]+'3'+'.png'


def obs_detect(depth_image, threshold = 500):
    left_flag = 0
    mid_flag = 0
    right_flag = 0    
    copy = depth_image.copy()
    second_smallest = sorted(list(set(copy.flatten().tolist())))[2]
    result = np.where(depth_image == second_smallest)
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
    