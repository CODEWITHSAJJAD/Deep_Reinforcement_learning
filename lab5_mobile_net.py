import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2,MobileNet,MobileNetV3Small,MobileNetV3Large
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import learning_phase
from tensorflow.python.keras.saving.saved_model.load import metrics
from tensorflow.python.layers.core import dropout


#####################################################
model1=MobileNet(
    input_shape=None, #(224,224,3)(32,32,3)
    alpha=1.0, #if alpha <1 them channel decreases and if if alpha >1 then channel increases in alpha is 1 then use those channels
    depth_multiplier=1.0,
    dropout=0.001,
    include_top=True, #false means you will add
    weights='imagenet',
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    name='bangali baba ka model',
)
x=model1.output
x=GlobalAveragePooling2D()(x)
x=Dense(128,activation='relu')(x)
predictions=Dense(4,activation='softmax')(x)
model=Model(input=model1.input,outputs=predictions)
for layer in model1.layers:
    layer.trainable=False
for layer in model1.layers[-5:]:
    layer.trainable=True

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain)

