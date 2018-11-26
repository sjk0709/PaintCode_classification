#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 16:38:27 2018

@author: song
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:49:19 2018

@author: song
"""

# -*- coding: utf-8 -*-
"""
Created on Tue April 3 10:56:53 2018

Convolutional VAriational Autoencode

@author: Jaekyung Song
"""

import time, sys, os
sys.path.append(os.pardir)  # parent directory
sys.path.append("../utils")
sys.path.insert(0, 'networks')

import numpy as np
#from xml.etree.ElementTree import Element, SubElement, dump, ElementTree
#import tensorflow as tf
import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.init as init
import torch.nn.functional as F
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pytorch_classification.utils import Bar, AverageMeter
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt


from resnet0 import resnet50 as Network
#from cnn0 import jkConv10 as Network

#import matplotlib.animation as animation
#from sklearn.feature_extraction import image
#from PIL import Image
#import glob
#import random

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)      
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    
def weights_init(m):
    classname = m.__class__.__name__
#            print(classname) 
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
#                m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0.0, 0.1)
        print("xavier_uniform")
        
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.normal_(0.0, 0.1)
        print("xavier_uniform")
#        m.weight.data.normal_(0.0, 0.1)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

class Commander :
    
    def __init__(self, args, data):
        
        self.args = args
        self.data = data
        
        self.featureC = self.args.feature_shape[0]
        self.featureH = self.args.feature_shape[1] 
        self.featureW = self.args.feature_shape[2]
             
        self.modelFilePath = "models/"
        if not os.path.exists(self.modelFilePath):
            os.mkdir(self.modelFilePath)           
        self.modelFilePath = self.modelFilePath + self.args.load_folder_file[0] + "/"
        if not os.path.exists(self.modelFilePath):
            os.mkdir(self.modelFilePath)  
#        self.modelFilePath = self.modelFilePath + self.args.load_folder_file[1] + "/"
#        if not os.path.exists(self.modelFilePath):
#            os.mkdir(self.modelFilePath)  
            
            
        self.resultFolderPath = "results/"
        if not os.path.exists(self.resultFolderPath):
            os.mkdir(self.resultFolderPath)            
            
        
        self.network = Network(self.args.feature_shape, self.args.label_size)        
#        self.network.apply(weights_init)  
            
        self.tempModelPath = self.modelFilePath + 'temp_all.pkl'
        self.saveModelPath = self.modelFilePath + self.args.save_folder_file[1] + '_all.pkl'
        self.saveParamsPath = self.modelFilePath + self.args.save_folder_file[1] + '_params.pkl'        
        self.loadModelPath = self.modelFilePath + self.args.load_folder_file[1] + '_all.pkl'
        self.loadParamsPath = self.modelFilePath + self.args.load_folder_file[1] + '_params.pkl'        
        print("Model path :", self.loadModelPath)
        
        if self.args.load_model:            
            try:                               
                self.network = torch.load(self.loadModelPath, map_location=lambda storage, location: storage)                
#                self.network.load_state_dict(torch.load(self.loadParamsPath))      # it loads only the model parameters (recommended)  
                print("\n--------" + self.args.load_folder_file[1] + " is restored--------\n")
                
            except:
                print("\n--------There are no models.--------\n")
                print("\n--------First learning.--------\n")
                pass
        else:         
            print("\n--------First learning.--------\n")
            
            
        if self.args.isGPU:
            self.network.cuda()
        else :
            self.network.cpu()
        
        self.Sigmoid = nn.Sigmoid()
        self.Relu = nn.ReLU()
        self.softmax = nn.Softmax()        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)


        self.last_activation = self.args.load_folder_file[0][-2]
        if self.last_activation =='S':
            print("We use the sigmoid function at the last step.")
            
        loss_type = self.args.load_folder_file[0][-1]
#        self.loss_func = nn.MSELoss()        
        self.loss_func = nn.CrossEntropyLoss()
        if loss_type =='B':
            self.loss_func = nn.BCELoss(size_average=False)
            print("We use the binary cross entropy loss.")
        elif loss_type == 'C':
            self.loss_func = nn.CrossEntropyLoss()
            print("We use the cross entropy loss.")
            
#        print(self.last_activation)        
#        print(loss_type)
        
    def recordTrainInformation(self, trainingEpochs, batchSize, minCost, maxAccuracy, elapsedTime):        
        note = Element("TrainingInformation")
        SubElement(note, "TrainingEpochs").text = str(trainingEpochs)
        SubElement(note, "BatchSize").text = str(batchSize)    
        SubElement(note, "MinCost").text = str(minCost)
        SubElement(note, "MaxAccuracy").text = str(maxAccuracy)
        SubElement(note, "ElapsedTime").text = str(elapsedTime)
#        dump(음표)                
#        ElementTree(음표).write(self.model_dir + "training_imformation.xml")
        
    def createModelInformationXML(self):        
        note = Element("ModelSetting")
        to = Element("ModelName")
        to.text = self._graphName    
        note.append(to)
#        SubElement(note, "FeatureWidth").text = str(self._featureShape[0])
#        SubElement(note, "FeatureHeight").text = str(self._featureShape[1])    
#        SubElement(note, "LabelSize").text = str(self._label_size)
#        dump(음표)                
#        ElementTree(음표).write(self.model_dir + self._graphName + ".xml")
#        
        
    def train(self, nReDataGeneration=1, nTrainingEpochs=20, batch_size=100):
             
        self.network.train()
                
#        minCost = 100000.
#        maxAccuracy = 0.
        elapsed_time = 0.
#        current_epoch = 0
        
        # train my model
        print('Learning Started!')      
        start_time = time.perf_counter()
        
        
        #############################################################3
       
        
        self.data.getPaintData(classNoList=args.classNoList, label_type='index', isTrain=False)
        
        #############################################################   
        for i in range(nReDataGeneration):
            
            eps_time = AverageMeter()
            bar = Bar('Training '+str(i), max=self.args.nTrainingEpochs)
            end = time.time()
                                      
            self.data.getPaintData(classNoList=args.classNoList, label_type='index', isTrain=True)
                
            Xnp = self.data.train.images
            Ynp = self.data.train.labels
            x = torch.from_numpy(Xnp.reshape([-1,self.featureC,self.featureW,self.featureH]))
            y = torch.from_numpy(Ynp)
            dataset = TensorDataset(x, y)        
            self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)  
            
            for epoch in range(1, nTrainingEpochs+1):
                for k, [image,label] in enumerate(self.train_loader):
                        
                    image = Variable(image)       
                    label = Variable(label)                            
                    
                    if self.args.isGPU:
                        image = image.cuda()
                        label = label.cuda()                   
                        
            
                    self.optimizer.zero_grad()
                    
                    output = self.network(image)
                    output = F.softmax(output, dim=1)
#                    print(output.size())
#                    print(label.size())
                    loss = self.loss_func(output, label)
                    
                    loss.backward()
                    self.optimizer.step()            
                   
                    # bookkeeping + plot progress
                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=epoch, maxeps=self.args.nTrainingEpochs, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
                
                if epoch % 10 == 0:                        
                    print('--------------------------------------------------------------------')
                    torch.save(self.network, self.saveModelPath)       
                    torch.save(self.network.state_dict(), self.saveParamsPath ) # It saves only the model parameters (recommended)     
                    torch.save(self.network, self.tempModelPath)       
                    
                    accuracy = commander.getCurrentAccuracy(batch_size=100, numIter=1, isTrainData=False)
                    self.network.train()                    
                    print('Epoch : %04d' % (i*nTrainingEpochs + epoch) )
                    print('Loss =', loss )
                    print('Accuracy =', accuracy) 
                    print('--------------------------------------------------------------------')                
                
            bar.finish()
       
        elapsed_time = (time.perf_counter() - start_time)
   
#        accuracy_train = self.getCurrentAccuracy(batch_size=self.args.batch_size, isTrainData=True)   
#        accuracy_test = self.getCurrentAccuracy(batch_size=self.args.batch_size, isTrainData=False)   
#        if args.isGPU:
#            accuracy_train = accuracy_train.cpu()  
#            accuracy_test = accuracy_test.cpu()
        
        print('=====================================================================')        
#        print("Accuracy for training data : %.1f" % (accuracy_train.data.numpy()[0]*100.), "%")
#        print("Accuracy for test data : %.1f" % (accuracy_test.data.numpy()[0]*100.), "%")
        print('Elapsed %.3f seconds.' % elapsed_time)
        print('%.0f h' % (elapsed_time/3600), '%.0f m' % ((elapsed_time%3600)/60) , '%.0f s' % (elapsed_time%60) )
        print('Learning Finished!')   
        print('=====================================================================')
        
    
    def getCurrentAccuracy(self, batch_size, numIter=1, isTrainData=False):
            
        self.network.eval()
                
        channel = args.feature_shape[0]
        height = args.feature_shape[1]
        width = args.feature_shape[2]        

        Xnp = self.data.test.images
        Ynp = self.data.test.labels
        if isTrainData:
            Xnp = self.data.train.images
            Ynp = self.data.train.labels
        
        x = torch.from_numpy(Xnp.reshape([-1,channel,height,width]))        
        y = torch.from_numpy(Ynp)        
        dataset = TensorDataset(x, y)        
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)           
        
        tester = iter(test_loader)
               
        avg_accuracy = 0.0
        for i in range(numIter):
            Xs_tensor, Ys_tensor = tester.next()
#            Y = Y.view(-1,1)
#            print(Y)
            Xs_var = Variable(Xs_tensor)
            Ys_var = Variable(Ys_tensor)
                
            if self.args.isGPU:
                Xs_var = Xs_var.cuda()
                Ys_var = Ys_var.cuda()
    
            output_var = self.network(Xs_var)
            output_var = F.softmax(output_var, dim=1)
            
            _, prediction = torch.max(output_var,1)
                
#            for k in range(batch_size):
#                plt.imshow(Xnp[k], cmap='gray')
#                plt.show()
#                plt.imsave( 'a.png', Xnp[k], cmap='gray')
#                print(prediction)
            
            accuracy = (prediction==Ys_var).double().sum()/batch_size            
#            print(accuracy)
            avg_accuracy += accuracy 
            
        avg_accuracy /= numIter
                
        return avg_accuracy
    
    
    
    def checkTrainingResult(self, nOutput=1, isResultSave=True):
         
        self.network.eval()
                
        channel = args.feature_shape[0]
        height = args.feature_shape[1]
        width = args.feature_shape[2]        

        
        Xnp = self.data.test.images
        Ynp = self.data.test.labels
        
                     
        
        x = torch.from_numpy(Xnp.reshape([-1,channel,height,width]))        
        y = torch.from_numpy(Ynp)        
        dataset = TensorDataset(data_tensor=x, target_tensor=y)        
        test_loader = DataLoader(dataset, batch_size=nOutput, shuffle=True)           
        

        tester = iter(test_loader)
        Xs_tensor, Ys_tensor = tester.next()
                  
        Xs_var = Variable(Xs_tensor)
        if self.args.isGPU:
            Xs_var = Xs_var.cuda()
   
        output_var, indices, pool_sizes = self.convNet(Xs_var, self.args.isGPU)
        output_var = self.deconvNet(output_var, indices, pool_sizes)
#        if self.last_activation=='S':
#            output = self.Sigmoid(output)
#            output = self.Softmax2d(output)

        if self.args.isGPU:
            Xs_var = Xs_var.cpu()
            output_var = output_var.cpu()
            
#        out_img = torch.squeeze(output.data)
        original_img = Xs_var.data.view(-1,height,height)        
        label_img = Ys_tensor.view(-1,height,width)
        outputSize = output_var.data.size()
        out_img = output_var.data.view(-1,outputSize[2],outputSize[3])
                      
        for i in range(nOutput):  
            originalImg = original_img[i].numpy()
#            print(originalImg)
            labelImg = label_img[i].numpy()
            outputImg = out_img[i].numpy()
            # a colormap and a normalization instance
            cmap = plt.cm.jet
            norm = plt.Normalize(vmin=outputImg.min(), vmax=outputImg.max())                
            # map the normalized data to colors
            resultImg = cmap(norm(outputImg))
            
            if isResultSave:
#                filePath = self.resultFolderPath + 'original%d.png' % i
#                plt.imsave(filePath, originalImg, cmap='gray')                
#                filePath = self.resultFolderPath + 'label%d.png' % i
#                plt.imsave(filePath, labelImg, cmap='gray')                
                filePath = self.resultFolderPath + 'output%d.png' % i
                plt.imsave(filePath, outputImg, cmap='gray')
                
                
#                from PIL import Image  
#                buffer = np.hstack((originalImg, resultImg)) 
#                filePath = self.resultFolderPath + 'compared_result%d.png' % i
#                plt.imsave(filePath, buffer, cmap='gray')
                
                filePath = self.resultFolderPath + 'heatmap%d.png' % i
                plt.imsave(filePath, resultImg)                
        
                
            plt.subplot(1,3,1)    
            plt.imshow(originalImg, cmap='gray') 
            plt.subplot(1,3,2)
            plt.imshow(outputImg, cmap='gray')
            plt.subplot(1,3,3)
            plt.imshow(resultImg)
            plt.show() 
                       
            print("Input size (H, W) :", originalImg.shape)
            print("Output size (H, W) :", (out_img.size()[1], out_img.size()[2]))
            print("========================================================")
            print("========================================================")
  





class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    

if __name__ == '__main__':  
    
    print("Torch version :", torch.__version__)
    
    # GPU check
    useGPU = torch.cuda.is_available()
    
    if useGPU :
        deviceNo = torch.cuda.current_device()
        print("GPU_is_available.")
        print("DeviceNo :",  deviceNo)
        print(torch.cuda.device(deviceNo))
        print("Device_count :", torch.cuda.device_count())
#        print(torch.cuda.get_device_name(0))
#        print("Device_capability :", torch.cuda.get_device_capability(deviceNo))
#        print("Device_max_memory :", torch.cuda.max_memory_allocated(deviceNo))
#        print("Device_max_memory_cached :", torch.cuda.max_memory_cached(deviceNo))
        
    else :
        print("There are no GPU.")
        
        
    args = dotdict({
            'dataPath' : "../../../DB_JK/PaintCode_dataset/PaintCode/" ,            
            'training' : True ,
            'isGPU' : False,         # False, # True,
            'load_model': True,      
            'load_folder_file': ("PC_resnet0_resnet50",'temp'), # ("ForClass4_jkfcn3/",'ForClass4_jkfcn3'), (DAGM_jkfcn3/, jkfcn3)            
            'save_folder_file': ("PC_resnet0_resnet50",'model0'), # ("ForClass4_jkfcn3/",'ForClass4_jkfcn3'), (DAGM_jkfcn3/, jkfcn3)            
            'classNoList' : [0,1,2,3,4,5,6,7,8,9],
            'feature_shape' : [1, 32, 32], # channel, H, W    # [1, 104, 113]
            'label_size' : 10,  # index 0 : something we don't know
            'nReDataExtraction' : 1,
            'nTrainingEpochs': 5000,
            'batch_size' : 128,            
            'learning_rate' : 1e-4,   
            })
    
       
    if useGPU==False and args.isGPU==True:
        args.isGPU = False
        print("GPU is not availabe.")
    if args.isGPU==False:
        print("Runing by CPU")
    
    import get_paintcode_data as PaintCode
    
    data = PaintCode.PaintCode(args.dataPath, feature_shape=args.feature_shape, feature_type='torch')    
    commander = Commander( args=args, data=data)    
    
    if(args.training==True):        	                
        commander.train( nReDataGeneration=args.nReDataExtraction, nTrainingEpochs=args.nTrainingEpochs, batch_size=args.batch_size)            
        
    elif(args.training==False):  
        commander.data.getPaintData(classNoList=args.classNoList, label_type='index', isTrain=False)

        accuracy = commander.getCurrentAccuracy(batch_size=100, numIter=1, isTrainData=False)   
        
        if args.isGPU:
            accuracy = accuracy.cpu()
        
        print('=====================================================================')        
#        print("Accuracy for training-data : %.1f" % (accuracy_train.data.numpy()[0]*100.), "%")
        print("Accuracy for test-data : %.1f" % (accuracy.data.numpy()[0]*100.), "%")
        print('=====================================================================')   

    