# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sys, os
sys.path.append(os.pardir)  # parent directory
#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import matplotlib.gridspec as gridspec
from sklearn.feature_extraction import image
# from PIL import Image
import cv2
import glob
import random
import struct


def genImgListWithFilename(folderpath, imgType, start, end): # input : path  # output : imgList   # path안의 이미지들을 리스트로 만들어준다.
    imgList = []    
    for i in range(start, end+1):
        filepath = folderpath+ '/' + str(i) + '.' + imgType                       
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # B, G, R               
#            cv2.imshow('ddd',image)
#            cv2.waitKey(0)
        imgList.append(image)    
    return imgList   

 
def cvRotateImg(img, angle):    
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1.0)
    return cv2.warpAffine(img,M,(cols,rows))




def getImages(path, format='png'): # input : path  # output : imgList   
    imgList = []
    for filepath in glob.glob(path + "/*."+format):    # make a image list with images in path
#        img = Image.open(filepath)          
#        keep = img.copy()
#        imgList.append(keep)    
#        img.close()              
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)  # B, G, R  cv2.IMREAD_COLOR  IMREAD_GRAYSCALE
        print(filepath, img.shape)          
#        cv2.imshow('ddd',img)
#        cv2.waitKey(0)
        imgList.append(img)
    return imgList



class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

if __name__ == '__main__':  
    
    settings = dotdict({
#            'dataPath' : '../../../JKcloud/DB_JK/DAGM2007_dataset',
            'dataPath' : '../../DB_JK/PaintCode_dataset/',
            'image_shape' : (104, 113, 1),
            'generated_image_folder' : 'Generated_Images2/',
            'feature_shape' : (32,32,1),
            })    
    
    
    temp_folder = settings.dataPath + "PaintCode/" 
    if not os.path.exists( temp_folder ):
        os.mkdir( temp_folder )
    train_dir = temp_folder + "Train/"
    if not os.path.exists(train_dir):
        os.mkdir( train_dir )
    test_dir = temp_folder + "Test/"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    for i in range(10):
        no_dir = train_dir + str(i)
        if not os.path.exists(no_dir):
            os.mkdir(no_dir)
        no_dir = test_dir + str(i)
        if not os.path.exists(no_dir):
            os.mkdir(no_dir)
    
    image_li = getImages(settings.dataPath, format='jpg')
    
    height = image_li[0].shape[0]
    width = image_li[0].shape[1]
    print("height :", height, "width :", width)
       
    
    temp_folder = "Cropped_original_images/"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
        
    cropedImg_li = []
    dh = int(width/4)
    for i, img in enumerate(image_li):
        for k in range(4):
            cropedImg = img[:, k*dh:(k+1)*dh]
            cropedImg = cropedImg[4:-10, 11:-12]           
            cropedImg = cv2.resize(cropedImg, (settings.feature_shape[0], settings.feature_shape[0]), interpolation=cv2.INTER_AREA)
#            cropedImg = cv2.resize(cropedImg, (settings.feature_shape[0], settings.feature_shape[1]), interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
#            cv2.imshow('ddd', cropedImg)
#            cv2.waitKey(0)    #아무키나 누르면 지나감 안에 값이 1이면 그냥 지나가지만 키를 눌렀을때 반응함
            cropedImg_li.append(cropedImg)

            cv2.imwrite(temp_folder + str(i) + "-" + str(4+k) + ".jpg", cropedImg)
    
    if not os.path.exists(settings.generated_image_folder):
        os.mkdir(settings.generated_image_folder)
   
    # one -> eight
    for i, img in enumerate(cropedImg_li):
        vertical_flip_img = cv2.flip(img,1)
        for k in range(0,4):
            augmentedImg1 = cvRotateImg(img, 90*k)
            augmentedImg2 = cvRotateImg(vertical_flip_img, 90*k)
            cv2.imwrite(settings.generated_image_folder + str(i) + "-" + str(90*k) + ".jpg", augmentedImg1)
            cv2.imwrite(settings.generated_image_folder + str(i) + "-flip-" + str(90*k) + ".jpg", augmentedImg2)
            
 

        