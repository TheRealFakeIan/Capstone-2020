#import init as init #has file paths
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import misc
import scipy.io as sio
import cv2
import numpy as np
import functions.helper_modules as hm



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

    #load data
    image_names = os.listdir(images_path)
    image_names.sort()
    ann_file = open(annotations_path)
    annotations = json.load(ann_file)


    #set up for first image
    cur_image_name = image_names[0]
    next_image_name = ''
    right_image_name = ''
    left_image_name = ''
    move_command = ''
    #fig,ax = plt.subplots(1)
    key = 0
    

    print("Running in autosearch mode")
    reach = False
    end = False
    cw_rotate = 0
    ccw_rotate = 0
    zoom = 0
    

    
    while not reach:
        #load the current image and annotations 
        rgb_image = cv2.imread(os.path.join(images_path,cur_image_name))
        #plot the image and draw the boxes
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 960, 540)
        cv2.imshow('image', rgb_image)
        
        key = cv2.waitKey(300)
        
        #Start off by moving forward
        next_image_name = annotations[cur_image_name]['forward']

        #If there is an image available, continue navigating forward
        if next_image_name != '':
            cur_image_name = next_image_name
            
        #STILL NEED WORK 
        elif next_image_name == '':
            rotate_count = 4
            for i in range(0, rotate_count):
                left_image_name = annotations[cur_image_name]['rotate_ccw']
                cur_image_name = left_image_name
                ccw_rotate+=1
            
            
                
                
            
            
        
        
        #Press Q to destroy windows and quit
        if key == 113:
            break
            cv2.destroyAllWindows()


    key = cv2.waitKey(-1)
    if key == 113:
        cv2.destroyAllWindows()
        
    
        
    
  
        
            



def vis_camera_pos_dirs(scene_path, plot_directions=True, scale_positions=True):
    """ Visualizes camera positions and directions in the scene.

    ARGUMENTS:
        scene_path: the string full path of the scene to view
            Ex) vis_camera_pos_dirs('/path/to/data/Home_01_1')

    KEYWORD ARGUMENTS:
        plot_directions: bool, whether or not to plot camera directions
                         defaults to True
            Ex) vis_camera_pos_dirs('Home_01_1', plot_directions=False)

        scale_positions: bool, whether or not to scale camera positions 
                         to be in millimeters. Defaults to True
            Ex) vis_camera_pos_dirs('Home_01_1', scale_positions=False)

    """

    #TODO - make faster - organize all positions/directions, then plot

    #set up scene specfic paths
    images_path = os.path.join(scene_path,'jpg_rgb')
    image_structs_path = os.path.join(scene_path,'image_structs.mat')

    #load data.
    #the transition from matlab to python is not pretty
    image_structs = sio.loadmat(image_structs_path)
    scale = image_structs['scale'][0][0]
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]

    #make plot
    fig,ax = plt.subplots(1)

    for camera in image_structs:
        #get 3D camera position in the reconstruction
        #coordinate frame. The scale is arbitrary
        world_pos = camera[3] 
        if scale_positions:
            world_pos *= scale

        #get 3D vector that indicates camera viewing direction
        #Add the world_pos to translate the vector from the origin
        #to the camera location.
        camera[4] /= 2;#to make plot look nicer
        if scale_positions:
            direction = world_pos + camera[4]*scale
        else:
            direction = world_pos + camera[4]
            
        #plot only 2D, as all camera heights are the same

        #draw the position
        plt.plot(world_pos[0], world_pos[2],'ro')    
        #draw the direction if user sets option 
        if plot_directions:
            plt.plot([world_pos[0], direction[0]], 
                             [world_pos[2], direction[2]], 'b-')    


    #for camera in image_structs 
    plt.axis('equal')
    plt.show()    
    