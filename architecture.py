from keras.models import Sequential
from keras.layers import *
from skimage.measure import compare_psnr
import numpy as np
import keras.backend as K
from keras.callbacks import *
def psnr(base_image,altered_image):
	try:
		MSE=K.mean(K.square(base_image-altered_image))
		if(MSE==0):
			return 100
		else:
			return 20*K.log(255.0/K.sqrt(MSE))
		
	except Exception as e:
		print(e)
		return K.constant(100)
def advance_relu(input_):
	return	K.relu( input_,alpha=0.0, max_value=255.0)

import pickle
bth=50
epc=1
model=Sequential()

mc=ModelCheckpoint("weights.{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
es=EarlyStopping(monitor='loss', min_delta=1, patience=3, verbose=1, mode='auto')
tb=TensorBoard(log_dir='./logs',  batch_size=bth, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
lr=ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


model.add(Conv2D(64,(3,3),input_shape=(256,256,3),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))


#conv2_1
model.add(Conv2D(128,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
#conv2_2

model.add(MaxPooling2D(pool_size=(2,2)))

#conv3_1
model.add(Conv2D(256,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
#conv3_2
model.add(MaxPooling2D(pool_size=(2,2)))
#CONV4_1
model.add(Conv2D(512,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

########################################
## DECONV STARTS 
########################################

#DECONV 1 -1
model.add(Conv2DTranspose(512,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(UpSampling2D(size=(2,2)))
# DECONV 2-1
model.add(Conv2DTranspose(256,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(UpSampling2D(size=(2,2)))

#DECONV 3-1
model.add(Conv2DTranspose(128,(2,2),strides=(1,1),padding="valid",
	activation="linear",
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(UpSampling2D(size=(2,2)))

# DECONV 4-1
model.add(Conv2DTranspose(3,(3,3),strides=(1,1),padding="valid",
	activation=advance_relu,
	kernel_initializer="truncated_normal",
bias_initializer="truncated_normal"))


model.summary()
k=model.get_config()
print(k)
print("values")
print(k.values)


model.compile(
loss="mse",
optimizer="adam",
metrics=['accuracy',psnr]
)
with open("./data.pkl",'rb') as f:
	data=pickle.load(f)	
model.fit(data['norain'].reshape(-1,256,256,3)[:100,:],data['rain'].reshape(-1,256,256,3)[:100,:],epochs=epc,batch_size=bth,shuffle=True,validation_split=0.3,callbacks=[mc,es,tb,lr])



