# -*- coding: utf-8 -*-
"""

"""
import sys, os
sys.path.append(os.pardir)  # parent directory
# --------------------------------------------------
#    PyQt5, exit code 1 (To catch the exceptions)
# ------------------------------------------------[1]
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook
# -----------------------------------------------[1]

import numpy as np
import cv2
#import math
#import random
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import time
#import glob
#import scipy.spatial as spatial

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

# -------------------------------
# 0. 프로그램 변수, 함수 정의
# -------------------------------
from skimage import io, color
#from sklearn.feature_extraction import image

import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
import torch.nn.functional as F
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
#from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
from torch.autograd import Variable
#from pytorch_classification.utils import Bar, AverageMeter

sys.path.insert(0, 'networks_pytorch')


ui, dummy = uic.loadUiType("paintCode.ui")



def conv_CV2_QPixmap(img):

    # ---------------------------------------------
    # Convert image: OpenCV -> QImage -> QPixmap
    # ---------------------------------------------
    # 1. Convert the image data from BGR to RGB
    img_CV2_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Notice the dimensions
    height, width, bytesPerComponent = img_CV2_RGB.shape
    bytesPerLine = bytesPerComponent * width

    # 3. Convert OpenCV(RGB) image to QImage
    img_Q = QImage(img_CV2_RGB.data, width, height, bytesPerLine, QImage.Format_RGB888)

    # 4. Convert QImage to QPixmap
    img_P = QPixmap.fromImage(img_Q)

    return img_P




 
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    

class myGuiProgram(QMainWindow, ui):
    def __init__(self, parent=None):   # 초기화 루틴
        super().__init__()
        self.setupUi(self)              # PyQT Layout (GUI) 셋업
        self.show()                     # GUI 보이기
  
    
        self.args = dotdict({
            'dataPath' : "../../../DB_JK/PaintCode_dataset/PaintCode/" ,            
            'training' : True ,
            'isGPU' : False,         # False, # True,
            'load_model': True,      
            'load_folder_file': ("PC_resnet0_resnet50/",'temp'), # ("ForClass4_jkfcn3/",'ForClass4_jkfcn3'), (DAGM_jkfcn3/, jkfcn3)            
            'save_folder_file': ("PC_resnet0_resnet50/",'model0'), # ("ForClass4_jkfcn3/",'ForClass4_jkfcn3'), (DAGM_jkfcn3/, jkfcn3)                        
            'classNoList' : [0,1,2,3,4,5,6,7,8,9],
            'feature_shape' : [1, 32, 32], # channel, H, W    # [1, 104, 113]
            'label_size' : 10,  # index 0 : something we don't know            
            'batch_size' : 100,       
            })
    
    
        self.pushButton_load.clicked.connect(self.clickPushButton_load)



    def readPaintCodeCNN_pytorch( self, args, inputs):
    
#        nImages = len(inputs)
#        channel = args.feature_shape.shape[0]
        height = args.feature_shape[1]
        width = args.feature_shape[2] 
        
        resultFolderPath = "./Results/"
        if not os.path.exists(resultFolderPath):
            os.mkdir(resultFolderPath) 
        modelPath = "./models_pytorch/"+ args.load_folder_file[0]  + args.load_folder_file[1] + '_all.pkl'
#        paramsPath = "networks_pytorch/"+ args.load_folder_file[0]  + args.load_folder_file[1] + '_params.pkl'
        
#        sigmoid = nn.Sigmoid()
#        softmax = nn.Softmax()
        
        network = ""
           
    #    print("Model path :", modelPath)
            
        try:        
            network = torch.load(modelPath, map_location=lambda storage, location: storage) 
            print("|=============================================|")
            print("|===== Starting to read paint_code by CNN =====|")
            print("|=============================================|")
            print( modelPath + " has been restored.")       
            
    #        print("Conv parameters path :", convParamsPath)
    #        network.load_state_dict(torch.load(paramsPath))      # it loads only the model parameters (recommended)  
    #        print("\n--------" + paramsPath + " is restored--------\n")           
            if args.isGPU:
                network.cuda()
            else :
                network.cpu()                                
            
        except:
            print("|================================|")
            print("|===== There are no models. =====|")
            print("|================================|")
            pass  
            
        network.eval()
        
        Xs = np.array(inputs, dtype=np.float32)/255.    
        Xs = Xs.reshape([-1,1,height,width])   
        Xs_tensor = torch.from_numpy(Xs)
        Xs_var = Variable(Xs_tensor)
        if args.isGPU:
            Xs_var = Xs_var.cuda()
          
        output_var = network(Xs_var)
        output_var = F.softmax(output_var, dim=1)
        
        
        if args.isGPU:
            Xs_var = Xs_var.cpu()
            output_var = output_var.cpu()
        
        probs, predictions = torch.max(output_var, 1)
        
        probs = probs.data.numpy()
        predictions = predictions.data.numpy()  
    
    
        print("|=======================================|")
        print("|===== Prediction is complete. =====|")
        print("|=======================================|")
          
        return probs, predictions 
    
    
       
    def imageToinputData(self, size, image): 
        cropedImg_li = [] 
#        print(image.shape)
        dh = int(image.shape[1]/4)
#        print(dh)
        for k in range(4):
            cropedImg = image[:, k*dh:(k+1)*dh]
            cropedImg = cropedImg[4:-10, 11:-12]          
            cropedImg = cv2.resize(cropedImg, (size[0], size[1]), interpolation=cv2.INTER_AREA)
#            cropedImg = cv2.resize(cropedImg, (settings.feature_shape[0], settings.feature_shape[1]), interpolation=cv2.INTER_CUBIC+cv2.INTER_LINEAR)
            cropedImg_li.append(cropedImg)
        
        return cropedImg_li
        
    def clickPushButton_load(self): 

        # ----------------------------------
        # 2. Get image file path, name
        file_path, _ = QFileDialog.getOpenFileName(None, "Open image file...")
#        folderPath = QFileDialog.getExistingDirectory(self, 'Select directory')
        self.folderPath = QFileInfo(file_path).absolutePath()
        
        if file_path == '':
            print('- No image selected')
            return
        else:
            print(file_path)      
    
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
#        cv2.imshow('ddd', img)
#        cv2.waitKey(0)
        img_P = conv_CV2_QPixmap(img)  
        flags = Qt.KeepAspectRatio
        img_P = img_P.scaled(self.label_cv2a.size(), flags)  # 라벨창 크기에 맞추기
        self.label_cv2a.setPixmap(img_P)
        
        c = self.args.feature_shape[0]
        h = self.args.feature_shape[1]
        w = self.args.feature_shape[2]        
        paintCodes = self.imageToinputData( (w, h), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))        
        paintCodes = np.array(paintCodes).reshape([-1, c , h, w] )             
        probs, numbers = self.readPaintCodeCNN_pytorch(self.args, paintCodes)
        
        self.label_No1.setText(str(numbers[0]))
        self.label_No2.setText(str(numbers[1]))
        self.label_No3.setText(str(numbers[2]))
        self.label_No4.setText(str(numbers[3]))
        self.label_prob1.setText(str(round(100*probs[0],1))+"%")
        self.label_prob2.setText(str(round(100*probs[1],1))+"%")
        self.label_prob3.setText(str(round(100*probs[2],1))+"%")
        self.label_prob4.setText(str(round(100*probs[3],1))+"%")
 
# ------------------------------------------
#    Python program Main routine
# ------------------------------------------
if __name__ == '__main__':

    # Create the application.
    app = QApplication(sys.argv)
    prog = myGuiProgram()      # 인스턴스 생성

    # sys.exit(app.exec_())
    try:
        sys.exit(app.exec_())  # Error 종료
    except:
        print("Exiting")       # 정상종료
# ------------------------------------------
