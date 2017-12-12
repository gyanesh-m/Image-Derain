import os
import pickle
from scipy import misc
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras.backend as K
from keras.callbacks import *
from keras import models

class Derain(object):
    def __init__(self, data_dir,checkpoint_dir='./checkpoints/'):
        self.data=None
        self.generate_data(data_dir)
        self.data_dir=data_dir
        self.bth=50
        self.epc=1
        self.mdl=None
        self.ckd=checkpoint_dir
        self.mc=ModelCheckpoint(checkpoint_dir+"weights.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        self.es=EarlyStopping(monitor='loss', min_delta=1, patience=15, verbose=1, mode='auto')
        self.tb=TensorBoard(log_dir='./logs',  batch_size=self.bth, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        self.lr=ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    def train(self, training_steps=10):
        """
            Trains the model on data given in path/train.csv
            	which conatins the RGB values of each pixel of the image  

            No return expected
        """
        self.mdl.fit(self.data['rain'].reshape(-1,256,256,3),self.data['norain'].reshape(-1,256,256,3),epochs=self.epc,batch_size=self.bth,shuffle=True,validation_split=0.1,callbacks=[self.mc,self.tb,self.es])

        
    #Not required here since we are using model checkpoint callback
    #So the call back saves it automatically after every epoch.
    def save_model(self, step):

        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """


    def load_model(self, **params):
    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
        files=os.listdir()
        file_name=[i for i  in files if '.hdf5' in i]
        print("#"*5+"  Using saved model-"+file_name[0]+"  "+"#"*5)
        model=models.load_model(os.path.join(os.getcwd(),file_name[0]),custom_objects={"psnr":self.psnr,"advance_relu":self.advance_relu})
        print("#"*5+"  Model Loaded  "+"#"*5)
        self.mdl=model


    def psnr(self,base_image,altered_image):
        try:
            MSE=K.mean(K.square(base_image-altered_image))
            if(MSE==0):
                return 100
            else:
                return 20*K.log(255.0/K.sqrt(MSE))/K.log(K.constant(10))
            
        except Exception as e:
            print(e)
            return K.constant(100)
    def advance_relu(self,input_):
        return  K.relu( input_,alpha=0.0, max_value=255.0)

    def generate_data(self,data_dir):
        print("#"*5+"  Resizing images  "+"#"*5)
        min_shape=(100000000,100000000)
        max_shape=(0,0)
        norain=[]
        rain=[]
        for j,i in enumerate(os.listdir(data_dir)):
            if('jpg' not in i.split('.')[-1]):continue
            ig=misc.imread(os.path.join(data_dir,i))
            print(i)
            norain_=ig[:,:ig.shape[1]//2,:]
            rain_=ig[:,ig.shape[1]//2:,:]
            if(norain_.shape[0]>=256 and norain_.shape[1]>=256):
                norain_=misc.imresize(norain_,((256,256,3)))
                rain_=misc.imresize(rain_,((256,256,3)))
            elif(norain_.shape[0]<256 and norain_.shape[1]>256):
                norain_=misc.imresize(norain_,((norain_.shape[0],256,3)))
                rain_=misc.imresize(rain_,((norain_.shape[0],256,3)))
                h=(256-norain_.shape[0]+1)//2
                w=0
                d=0
                norain_=np.pad(norain_,((h,h),(w,w),(d,d)),"constant")
                rain_=np.pad(rain_,((h,h),(w,w),(d,d)),"constant")
                norain_=misc.imresize(norain_,((256,256,3)))
                rain_=misc.imresize(rain_,((256,256,3)))           
            elif(norain_.shape[0]>256 and norain_.shape[1]<256):
                norain_=misc.imresize(norain_,((256,norain_.shape[1],3)))
                rain_=misc.imresize(rain_,((256,norain_.shape[1],3)))
                h=0
                w=(256-norain_.shape[1]+1)//2
                d=0
                norain_=np.pad(norain_,((h,h),(w,w),(d,d)),"constant")
                rain_=np.pad(rain_,((h,h),(w,w),(d,d)),"constant")
                norain_=misc.imresize(norain_,((256,256,3)))
                rain_=misc.imresize(rain_,((256,256,3)))       
            else:
                h=(256-norain_.shape[0]+1)//2
                w=(256-norain_.shape[1]+1)//2
                d=(3-norain_.shape[2]+1)//2
                norain_=np.pad(norain_,((h,h),(w,w),(d,d)),"constant")
                rain_=np.pad(rain_,((h,h),(w,w),(d,d)),"constant")
                norain_=misc.imresize(norain_,((256,256,3)))
                rain_=misc.imresize(rain_,((256,256,3)))           
            norain.append(norain_)
            rain.append(rain_)

        self.data={"norain":np.array(norain),
                "rain":np.array(rain)
              }
        print("#"*5+"  Creating directory for checkpoint  "+"#"*5)
        try:
            os.mkdir(self.ckd)
        except:
            pass
    def test(self):
        print("#"*5+"  Testing  "+"#"*5)
        loss_metrics=self.mdl.evaluate(self.data['rain'].reshape(-1,256,256,3),self.data['norain'].reshape(-1,256,256,3),batch_size=1)
        count=0
        
        
        for i in loss_metrics:
            print(self.mdl.metrics_names[count],i)
            count+=1

#specify the path of the test case dataset as a parameter to the Derain class.
d=Derain(os.path.join(os.getcwd(),"testcases/"))
#loads saved model
d.load_model()
#tests on the saved model
d.test()