#!/usr/bin/env python

import sys, os
sys.path.append(os.pardir)  # parent directory
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction import image
# from PIL import Image
import cv2
import glob
import random
import struct


# PIL_JK class includes PIL util made by JK

class Data(object):
    def __init__(self):
        self.images = np.zeros(1)
        self.labels = np.zeros(1)
        self.start_batch = 0
        self.end_batch = 0
        self.num_examples = 0
        
    def next_batch(self, batch_size):
        mini_batch = np.random.choice(len(self.images), batch_size, replace=False)
        
#        self.end_batch = self.start_batch+batch_size
#        mini_batch = np.arange(self.start_batch, self.end_batch)
#        if self.end_batch!=len(self.images):
#            self.start_batch = self.end_batch
#        else :
#            self.start_batch = 0
            
        return self.images[mini_batch], self.labels[mini_batch]
              

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
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(img,M,(cols,rows))
    return image
   

# data augmentation                  
def dataAugmentation(image):
    Xli = []
    
    verticalFlip = cv2.flip(image,1)         # vertical flip                                 
    for i in range(1, 5):                    
        augmentedImg1 = cvRotateImg(image, 90*i)
        augmentedImg2 = cvRotateImg(verticalFlip, 90*i)
        Xli.append(augmentedImg1)        
        Xli.append(augmentedImg2)        
        
    return Xli





class PaintCode(object):
    """
    
    """
    def __init__(self, dataPath, feature_shape, feature_type='torch'):
        
        
        self.dataPath = dataPath
          
        self.feature_shape = feature_shape  
        self.height = feature_shape[0]
        self.width = feature_shape[1]
        self.channel = feature_shape[2]
        if feature_type=='torch':
            self.channel = feature_shape[0]
            self.height = feature_shape[1]
            self.width = feature_shape[2]

        #readFreeImg()
        self.train = Data()
        self.test = Data()        

    
        
    def getImages(self, path, size,  format='png'): # input : path  # output : imgList   
        imgList = []
        for filepath in glob.glob(path + "/*."+format):    # make a image list with images in path
    #        img = Image.open(filepath)          
    #        keep = img.copy()
    #        imgList.append(keep)    
    #        img.close()
    #        print(filepath)        
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # B, G, R  cv2.IMREAD_COLOR 
#            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
#            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
#            cv2.imshow('ddd',img)
#            cv2.waitKey(0)        
            imgList.append(img)
        return imgList
    
     
    def getPaintData(self, classNoList=[0,1,2,3,4,5,6,7,8,9], label_type='index', isTrain=True):
        
        dataX = []
        dataY = []
        
                        
        for classNo in classNoList:
            class_path = ''
            if isTrain :
                class_path = self.dataPath + 'Train/' + str(classNo)         
            else :
                class_path = self.dataPath + 'Test/' + str(classNo) 
                
#            print(class_path)
            tempX = self.getImages(class_path, (self.width, self.height), format='jpg')                    
            dataX += tempX
            for k in range(len(tempX)):
                if label_type=='image' or label_type=='array':
                    label = np.zeros(10)
                    label[classNo] = 1
                    dataY.append(label) 
                elif label_type=='index':
                    dataY.append(classNo)         
            
        self.label_type = label_type   # image, index, or array
                           
#        cv2.imshow('ddd',dataX[10])
#        cv2.waitKey(0) 
#        print("ddddddddddddddddddddd", dataY[10])
        dataX = np.array(dataX, dtype=np.float32)/255.        
        dataX = dataX.reshape([-1, self.feature_shape[0], self.feature_shape[1], self.feature_shape[2]])
        
        if label_type=='image' or label_type=='array':
            dataY = np.array(dataY, dtype=np.float32)
        elif label_type=='index':
            dataY = np.array(dataY, dtype=np.int64)
        
        if isTrain:
            self.train.images = dataX
            self.train.labels = dataY
            self.train.num_examples = self.train.images.shape[0]
            print("Train images shape :", self.train.images.shape)
            print("Train labels shape :", self.train.labels.shape)
        else :
            self.test.images = dataX
            self.test.labels = dataY
            self.test.num_examples = self.test.images.shape[0]
            print("Test images shape :", self.test.images.shape)
            print("Test labels shape :", self.test.labels.shape)
 
    
      
    
if __name__ == '__main__':  
    

    dataPath = "../../../DB_JK/PaintCode_dataset/PaintCode/"  

        
    dagm = PaintNumber(dataPath)  
    dagm.getBlockImages(blockW=256, blockH=256, nOKperClass=1, nNGperClass=1, isTrain=False)
#    dagm.getFullImages( sizeW=300, sizeH=300, nOKperClass=1, nNGperClass=1, isTrain=False)
#    print(dagm.test.images[700])
    
    temp = dagm.test.images[0]
    plt.imshow(temp, cmap='gray')
    plt.show() 
    temp = dagm.test.labels[0]
    plt.imshow(temp, cmap='gray')
    plt.show() 

#    cv2.imshow('ddd', dagm.train.images[0])
#    cv2.waitKey(0)