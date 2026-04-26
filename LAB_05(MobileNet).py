import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, MobileNet, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers.agents.evaluate_agent import classifier

# parameters for model creation:
# 1. by default:
model1 = MobileNet(
                   input_shape=None, # by default size=(224,224,3), (32,32,3)
                   alpha=1.0, #1=channel equal to mobilenet, >1=channels inc every time, <alpha=dec everytime
                   depth_multiplier=1.0, # default=1.0, >1=resolution inc, <1=resolution dec
                   dropout=0.001, # kill neurons in specific layer, so that model will not overfit
                   include_top=True, # true means top 2 layers include, false means add top layers by yourself, neurons=no. of classes
                   weights='imagenet',  # none=it learns by itself, all weights randomly initialize
                   input_tensor=None, # none= image process, true=csv file
                   pooling=None,# bydefault it is none, 'avg',''max'
                   classes=1000, # 1000 default, but for our data it depend on our classes -> 4
                   classifier_activation='softmax', # sigmoid for 2 classes, softmax for more than 2
                   name=None # name of model
)

# 2. our customized model which is using mobilenet features:
x = model1.output
x = GlobalAveragePooling2D()(x) # add pooling layer and save back to x
x = Dense(128,activation='relu')(x) # x as input
predictions = Dense(4,activation='softmax')(x) # 4 nmbr of classes
#  model: take input as mobilenet input and output of predicted
model = Model(inputs=model1.input,outputs=predictions)

for layer in model1.layers:  # all layers training false
    layer.trainable = False # take an input layer and make depthwise training

for layer in model1.layers[-5:]: # include only 5 layers of mobilenet
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtrain, Ytrain) # fit on our own data

# Version 2:
model2 = MobileNetV3Small(
                   input_shape=None, # by default size=(224,224,3), (32,32,3)
                   alpha=1.0, #1=channel equal to mobilenet, >1=channels inc every time, <alpha=dec everytime
                   depth_multiplier=1.0, # default=1.0, >1=resolution inc, <1=resolution dec
                   dropout=0.001, # kill neurons in specific layer, so that model will not overfit
                   include_top=True, # true means top 2 layers include, false means add top layers by yourself, neurons=no. of classes
                   weights='imagenet',  # none=it learns by itself, all weights randomly initialize
                   input_tensor=None, # none= image process, true=csv file
                   pooling=None,# bydefault it is none, 'avg',''max'
                   classes=1000, # 1000 default, but for our data it depend on our classes -> 4
                   classifier_activation='softmax', # sigmoid for 2 classes, softmax for more than 2
                   name=None # name of model
                   minimalistic=,
                   include_preprocessing=
)
