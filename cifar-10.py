import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Activation,BatchNormalization

import pandas as pd
import numpy as np


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

size=32*32*3
x_train=x_train.reshape(-1,size)/255.0
x_test=x_test.reshape(-1,size)/255.0


model=Sequential([
Conv2D(32,(3,3),Activation('relu'),input_shape=(size,),padding='same'),
MaxPooling2D(pool_size=(2,2)),
Conv2D(32,(3,3), Activation('relu'),padding='same'),
MaxPooling2D(pool_size=(2,2)),
Conv2D(32,(3,3), Activation('relu'),padding='same'),
MaxPooling2D(pool_size=(2,2)),
Conv2D(32,(3,3), Activation('relu'),padding='same'),
MaxPooling2D(pool_size=(2,2)),
Flatten(),

Dense(size,Activation('relu'),name="Inputs"),
Dropout(0.1),
Dense(size/2,Activation('relu'),name="Hidden_Layer1"),
Dropout(0.1),
Dense(size/4,Activation('relu'),name="Hidden_Layer2"),
Dropout(0.1),
Dense(10,Activation('softmax'),name="Output")
])

optimizer=keras.optimizers.Adadelta(lr=0.01)

model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,epochs=10,validation_split=0.1,validation_data=(x_test,y_test),batch_size=64,verbose=1)
loss,acc=model.evaluate(x_test,y_test)
print("Accuracy:%{:.2f},Loss:{:.2f}".format(100*acc,loss))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


for i in range(30):
	prediction=model.predict(x_test[i].reshape(-1,size))
	print("Label: ",classes[y_test[i][0]],"Prediction: ",classes[prediction.argmax()])