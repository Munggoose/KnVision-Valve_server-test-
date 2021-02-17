import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


'''
Create blended heat map with JET colormap 
'''

# def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
def create_heatmap(im_map, im_cloud, colormap=cv2.COLORMAP_HOT, a1=0.5, a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud, colormap)
    # im_map = im_map + im_cloud
    im_map = im_map + im_cloud_clr
    return im_map
    
    
def calc_diff(real_img, generated_img, batchsize, thres=0.67): # 0.7 for multi batch 0.67
# def calc_diff(real_img, generated_img, batchsize, thres=10):
    diff_img = real_img - generated_img

    ch3_diff_img =  diff_img
    
    if batchsize == 1:
        diff_img = np.sum(diff_img, axis=0)
    else:
        diff_img = np.sum(diff_img, axis=1)
    diff_img = np.abs(diff_img)
    
    diff_img = np.log(diff_img + 1.5)
    if batchsize == 1:
        diff_img[diff_img < thres] = 0.0
    else:
        for bts in diff_img:
            bts[bts <= thres] = 0.0
        
    diff_img = cv2.normalize(diff_img, diff_img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return diff_img, ch3_diff_img
