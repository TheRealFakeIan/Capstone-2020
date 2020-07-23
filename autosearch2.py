# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:03:38 2020

@author: timot
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2 as cv
import numpy as np
import test_modules as tm


root = 'C:/Capstone2020/ActiveVisionDataset/'

if len(sys.argv) > 1:
    scene_name = sys.argv[1]
    scene_path = os.path.join(root,scene_name)
    requested_instance = sys.argv[2]

instance_found = tm.check_present_instance(scene_path, requested_instance)
if not instance_found:
    print("Instance requested is not in the selected scene.")
    
else:
    #VISUALIZATIONS
    name_to_id_dict = tm.get_instance_name_to_id_dict(root)
    id_num = name_to_id_dict[requested_instance[:]]

    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    annotations_path = os.path.join(scene_path,'annotations.json')

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)

    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = '' 
    move_command = ''
    
    fig,ax = plt.subplots(1)
    approaches = 0
    turned = 0
    found = False
    reach = False
    dead_end = False
    looped = False
    first_dead_end = ''
    
    while not reach:
        img = tm.plot_image(images_path, cur_image_name, annotations, ax)
        boxes = annotations[cur_image_name]['bounding_boxes']
        found = tm.search(id_num, boxes, img, ax)
        
        if approaches >=1:
            next_image_name = tm.command('d', cur_image_name, annotations)
            approaches = 0
     
        
        if found:    
            reach = tm.approach(images_path, cur_image_name, annotations, ax, id_num)
            approaches+=1
            
        else:      
            if dead_end:
                if turned >= 12:
                    next_image_name = tm.command('w', cur_image_name, annotations)
                    turned = 0
                    dead_end = False
                else:
                    next_image_name = tm.command('d', cur_image_name, annotations)
                    turned+=1
                    
            elif looped:
                if turned >= 12:
                    next_image_name = tm.command('w', cur_image_name, annotations)
                    turned = 0
                    looped = False
                else:
                    next_image_name = tm.command('a', cur_image_name, annotations)
                    turned+=1
                    
            else:
                next_image_name = tm.command('w', cur_image_name, annotations)
                

                  
        if next_image_name != '':
            if dead_end:
                next_image_name = tm.command('d', cur_image_name, annotations)
                cur_image_name = next_image_name
                turned+=1
            else:   
                cur_image_name = next_image_name
                
        elif next_image_name == '':     
            
            if cur_image_name == first_dead_end:
                looped = True
                next_image_name = tm.command('a', cur_image_name, annotations) 
                cur_image_name = next_image_name
                first_dead_end = ''
                turned+=1
                dead_end = False
                
            else:
                looped = False
                if first_dead_end == '':
                    first_dead_end = cur_image_name
                    next_image_name = tm.command('d', cur_image_name, annotations) 
                    cur_image_name = next_image_name
                    turned+=1
                    dead_end = True
                else:
                    next_image_name = tm.command('d', cur_image_name, annotations) 
                    cur_image_name = next_image_name
                    turned+=1
                    dead_end = True
                
                
        
    plt.waitforbuttonpress(-1)
                    
                
        
