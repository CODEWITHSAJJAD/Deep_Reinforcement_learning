# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing  import MinMaxScalar
# import keras
# import tensorflow as tf
import time
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

#-----------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('heart.csv')
X = data.drop(['output'],axis=1)
Y = data['output']
print("X= ",X.shape)
print("Y= ",Y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

hiddenlist=[]
timelist=[]
acclist=[]

st=time.time()
NNmodel = Sequential()
hiddenlist.append(1)
NNmodel.add(Dense(units=65,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary1=NNmodel.fit(X_train,Y_train,epochs=1000,batch_size=100,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])   # 83
acclist.append(result[1])

st=time.time()
NNmodel = Sequential()
hiddenlist.append(2)
NNmodel.add(Dense(units=60,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary2=NNmodel.fit(X_train,Y_train,epochs=1500,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])   # 85
acclist.append(result[1])

st=time.time()
NNmodel = Sequential()
hiddenlist.append(3)
NNmodel.add(Dense(units=50,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary3=NNmodel.fit(X_train,Y_train,epochs=2000,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])    # 83
acclist.append(result[1])


st=time.time()
NNmodel = Sequential()
hiddenlist.append(4)
NNmodel.add(Dense(units=40,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary4=NNmodel.fit(X_train,Y_train,epochs=2000,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])  # 86
acclist.append(result[1])


st=time.time()
NNmodel = Sequential()
hiddenlist.append(5)
NNmodel.add(Dense(units=30,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary5=NNmodel.fit(X_train,Y_train,epochs=2000,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])   # 85
acclist.append(result[1])

st=time.time()
NNmodel = Sequential()
hiddenlist.append(6)
NNmodel.add(Dense(units=16,activation='relu',input_shape=(13,)))
NNmodel.add(Dense(1,activation='sigmoid'))
op = keras.optimizers.Adadelta(learning_rate=0.01)
NNmodel.compile(loss='binary_crossentropy',optimizer=op,metrics=['accuracy'])
summary6=NNmodel.fit(X_train,Y_train,epochs=2000,batch_size=100,verbose=1)
result = NNmodel.evaluate(X_test,Y_test)
et = time.time()
etime = et-st
timelist.append(etime)
print("LOSS: ",result[0])
print("ACCURACY: ",result[1])  # 86
acclist.append(result[1])


# print("Hidden layers/t"+"Time/t"+"Accuracy")
# for i in range(4):
#     print(hiddenlist[i],"/t",timelist[i],"/t"+acclist[i])