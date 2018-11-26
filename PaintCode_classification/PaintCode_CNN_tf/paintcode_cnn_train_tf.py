# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:56:53 2017

@author: Jaekyung
"""
import sys, os
sys.path.append(os.pardir)  # parent directory
sys.path.append("../utils")
sys.path.insert(0, 'networks')
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import time
from xml.etree.ElementTree import Element, SubElement, dump, ElementTree

import jkcnn0 as Network

tf.set_random_seed(777) # reproducibility

class Commander :
    
    def __init__(self, data, Model, args):
        
        self.args = args

        #=====
        self.checkpoint_state_name = "checkpoint_state"
        self.saved_checkpoint = 'saved_checkpoint'
        self.input_graph_name = "input_graph.pb"              
        
        
        # initialize
        self.sess = tf.Session()        
        self._model = Model(self.sess, name="PaintNo", learning_rate=self.args.learning_rate, 
                                     feature_shape=self.args.feature_shape, lable_size=self.args.label_size )
        
        self.data = data
        
        self.model_dir = "models/"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)           
        self.save_dir = self.model_dir + self.args.save_folder_file[0]
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)      
            
        
        self.checkpoint_state = "checkpoint_state"
        self.save_model_path = self.save_dir + self.args.save_folder_file[1]
        self.optimal_model_path = self.save_dir + "optimal"
                
        self.load_dir = self.model_dir + self.args.load_folder_file[0]
        self.load_model_path = self.load_dir + self.args.load_folder_file[1] + ".meta"
       
        self.saver = tf.train.Saver()        
        checkpoint = tf.train.get_checkpoint_state(self.load_dir, latest_filename=self.checkpoint_state)

        # graph
        self.input_graph_path = self.model_dir + self.input_graph_name         
        self.checkpoint_prefix = self.save_dir + self.saved_checkpoint
        self.input_checkpoint_path = self.checkpoint_prefix + "-0"  


        self.sess.run(tf.global_variables_initializer())
        
        if self.args.load_model:
            if checkpoint and checkpoint.model_checkpoint_path:
                print(checkpoint)
                print(checkpoint.model_checkpoint_path)
    #            self.saver = tf.train.import_meta_graph(self.load_model_path)
    #            self.saver.restore(self.sess,tf.train.latest_checkpoint('./'))
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("%s has been loaded." % checkpoint.model_checkpoint_path)        
        else :
            print("First learning.")

       
        
        
    def recordTrainInformation(self, trainingEpochs, batchSize, minCost, maxAccuracy, elapsedTime):        
        note = Element("TrainingInformation")
        SubElement(note, "TrainingEpochs").text = str(trainingEpochs)
        SubElement(note, "BatchSize").text = str(batchSize)    
        SubElement(note, "MinCost").text = str(minCost)
        SubElement(note, "MaxAccuracy").text = str(maxAccuracy)
        SubElement(note, "ElapsedTime").text = str(elapsedTime)
        dump(note)                
        ElementTree(note).write(self.model_dir + "training_imformation.xml")
        
    def createModelInformationXML(self):        
        note = Element("ModelSetting")
        to = Element("ModelName")
        to.text = self.args.load_folder_file[1]    
        note.append(to)
        SubElement(note, "FeatureWidth").text = str(self.args.feature_shape[0])
        SubElement(note, "FeatureHeight").text = str(self.args.feature_shape[1])    
        SubElement(note, "LabelSize").text = str(self.args.label_size)
        dump(note)                
        ElementTree(note).write(self.model_dir + self.args.load_folder_file[1] + ".xml")
        
        
    def train(self, nReDataExtraction=5, training_epochs=20, batch_size=128):
        # Save our model
        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)
            
        start_time = time.perf_counter()
        minCost = 100000.
        maxAccuracy = 0.
        elapsed_time = 0.
        current_epoch = 0
        
        # train my model
        print('Learning Started!')        
        
        self.data.getPaintData(classNoList=self.args.classNoList, label_type='array', isTrain=False)
        for i in range(nReDataExtraction):                             
            self.data.getPaintData(classNoList=self.args.classNoList, label_type='array', isTrain=True)
            
            for epoch in range(training_epochs):
                avg_cost = 0.0
                total_batch = int(self.data.train.num_examples / batch_size)                               
                
                for k in range(total_batch):
                    batch_xs, batch_ys = self.data.train.next_batch(batch_size)
                #                print(batch_xs.shape, batch_ys.shape)
                    # train each model                
                    cost, _ = self._model.train(batch_xs, batch_ys)
                    avg_cost += cost            
                                
                avg_cost /= total_batch 
                
                if epoch % 50 == 0:
                    # save parameters, training information and our model 
                    #            save_path = saver.save(sess, checkpoint_path + '/network')
                    temp_accuracy = self.test()
                    minCost = min(minCost, avg_cost)
                    maxAccuracy = max(maxAccuracy, temp_accuracy)
                    
                    save_path = self.saver.save(self.sess, self.save_model_path, global_step=0, latest_filename=self.checkpoint_state)
                    save_path = self.saver.save(self.sess, self.optimal_model_path)                                                                    
                    print('--------------------------------------------------------------------')
                    print('Current model has been saved.')
                    print('Epoch : %04d' % (i*training_epochs + epoch), ' |  Cost =', avg_cost) 
                    print('--------------------------------------------------------------------')
            
            
        # show all variables name
#        for op in tf.get_default_graph().get_operations():
#            print (str(op.name))
        
        elapsed_time = (time.perf_counter() - start_time)
         
        # Save training information and our model                 
        self.recordTrainInformation(current_epoch, batch_size, minCost, maxAccuracy, elapsed_time)
        tf.train.write_graph(self.sess.graph_def, self.model_dir, self.input_graph_name, as_text=True)        
            
        print('=====================================================================')
        print('Minimum cost : ', minCost)
        print("Maximum accuracy : ", maxAccuracy)
        print('Elapsed %.3f seconds.' % elapsed_time)
        print('%.0f h' % (elapsed_time/3600), '%.0f m' % ((elapsed_time%3600)/60) , '%.0f s' % (elapsed_time%60) )
        print('Learning Finished!')   
        print('=====================================================================')
        
        
        
        
    def test(self, testBatchSize=50):
        
        # Test model and check accuracy               
        avg_accuracy = 0.0
        total_batch = int(self.data.test.num_examples / testBatchSize)          
        
        for i in range(total_batch):
            testX, testY = self.data.test.next_batch(testBatchSize)         
            accuracy = self._model.get_accuracy(testX, testY)
            avg_accuracy += accuracy 
        
        avg_accuracy /= total_batch
#        print('logits : ', self._model.predict(testX))     
        print('--------------------------------------------------------------------')
        print('Accuracy: %.2f' %(avg_accuracy*100.0), "%")
        print('--------------------------------------------------------------------')
        
        return avg_accuracy
        
    
    def freezeModel(self, output_node_names="prob" ):                   
        # Note that we this normally should be only "output_node"!!!
        input_saver_def_path = "" 
        input_binary = False                        
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        input_graph_path = self.model_dir + self.input_graph_name 
        output_graph_path = self.model_dir + self.args.load_folder_file[1] + ".pb"
        clear_devices = False        
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, self.input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_path,
                                  clear_devices, False)
        
        # make XML
        self.createModelInformationXML()
        print('Freezing the model finished!')
        
        
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
       
            
if __name__ == '__main__':  
       
    print("Tensorflow version :", tf.__version__)

    args = dotdict({
            'dataPath' : "../../../DB_JK/PaintCode_dataset/PaintCode/",
            'training' : True ,
            'load_model': True,     
            'save_folder_file' : ("PaintCode_jkcnn0_cpu_32/", 'checkpoint0'),
            'load_folder_file' : ("PaintCode_jkcnn0_cpu_32/", 'checkpoint0-0'), 
            'classNoList' : [0,1,2,3,4,5,6,7,8,9],
            'feature_shape' : [32, 32, 1], # channel, H, W    # [1, 104, 113]
            'label_size' : 10,  # index 0 : something we don't know
            'nReDataExtraction' : 1,
            'nTrainingEpochs': 5000,
            'batch_size' : 128,            
            'learning_rate' : 1e-4,                       
            })
    
    
    import get_paintcode_data as PaintCode    
    data = PaintCode.PaintCode(args.dataPath, feature_shape=args.feature_shape, feature_type='tf')  
    commander = Commander(data=data, Model=Network.Model, args=args)    
#    
    if(args.training==True):
        commander.train(nReDataExtraction=args.nReDataExtraction, training_epochs=args.nTrainingEpochs, batch_size=args.batch_size)    
        commander.freezeModel()
        
    elif(args.training==False): 
#        mode = int(input("1.training  |  2.accuracy test  |  3.Freeze a model  : "))
        data.getPaintData(classNoList=args.classNoList, label_type='index', isTrain=False)
        commander.test(50) 

