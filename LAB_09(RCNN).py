import cv2
import matplotlib.pyplot as plt
import numpy as np
from LAB_08_NMS import NMS, IoU
from LAB_10_ReadData import load_data
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model, Sequential

# make sequential model by your dataset
input_shape = (128,128,3)
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=256,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(Conv2D(filters=256,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(rate=0.35)) # drop bcz too many layers
model.add(Dense(1,activation='sigmoid'))
X,Y = load_data()

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
op = Adam(learning_rate=0.08)
model.compile(optimizer=op, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train,Y_train, epochs=8)
print(model.evaluate(X_test,Y_test))
model.save('Base_model.keras')

# R_CNN major code:

def rcnn(image, model):   # create for prediction
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # create selective object
    ss.setBaseImage(image)   # to generate region of intrest
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    copy = image.copy()  # region generate hogy
    copy2 = image.copy()  # actual display krny k liyay
    positive_boxes = []
    probs = []
    for box in results:  # some regions are of work
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]+box[2]
        y2 = box[1]+box[3]

        roi = image.copy()[y1:y2, x1:x2] # pick some pixels of copied image
        roi = cv2.resize(roi,(128,128)) # roi ka size same kr diya
        roi_use = roi.reshape((1,128,128,3)) # 1 sample/image, 128 by 128 size, 3 channels
        class_pred = model.predict(roi_use)[0][0] # model which we make

        if class_pred == 1:
            prob = model.predict(roi_use)[0][0]
            if prob > 0.98:
                positive_boxes.append([x1,y1,x2,y2])
                probs.append(prob)
                cv2.rectangle(copy2, (x1,y1),(x2,y2),(255,0,0),5)

    cleaned_boxes = NMS(np.array(positive_boxes),0.1)
    total_boxes = 0
    for clean_box in cleaned_boxes:
        clean_x1 = clean_box[0]
        clean_y1 = clean_box[1]
        clean_x2 = clean_box[2]
        clean_y2 = clean_box[3]
        total_boxes+=1
        cv2.rectangle(copy,(clean_x1,clean_y1),(clean_x2,clean_y2),(0,255,0),3)
    plt.imshow(copy)
    plt.show()

def main(image_name,model_name):
    test_img = cv2.imread(image_name)
    rcnn(test_img,model_name)

rcnn_model = (tf.keras.models.load_model('Base_model.keras'))
main('test_img1.jpg',rcnn_model)