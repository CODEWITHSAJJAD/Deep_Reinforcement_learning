import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import time
data=pd.read_csv('heart.csv')
x=data.drop(['output'],axis=1)
y=data['output']
print("x=",x.shape)
print("y=",y.shape)
############################################################
sc=MinMaxScaler()
x1=sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=0)
st=time.time()
nnmodel=Sequential()
nnmodel.add(Dense(units=20,activation='relu',input_shape=(13,)))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))
nnmodel.add(Dense(units=10, activation='relu'))

nnmodel.add(Dense(1,activation='sigmoid',))
op=keras.optimizers.Adam(learning_rate=0.002)
nnmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])#categorical_crossentropy
summary=nnmodel.fit(x_train,y_train,epochs=300,verbose=1)
result=nnmodel.evaluate(x_test,y_test)
et=time.time()
elapsed_time=et-st
print("Loss =",result[0])
print("Accuracy =",result[1])
#######################################################3
prediction=nnmodel.predict(x_test)
print("Predictions =",prediction)
prediction=tf.reshape(prediction,[-1])
prediction=[1 if i >=0.5 else 0 for i in prediction ]
from sklearn.metrics import  accuracy_score
acc=accuracy_score(y_test,prediction)
print("testing accuracy=",acc)
pd.DataFrame(summary.history).plot(figsize=(10,6))
plt.show()