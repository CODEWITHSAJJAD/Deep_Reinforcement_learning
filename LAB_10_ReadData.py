# read pictures dataset and make bounding boxes and seperate class
# object detection lab : read and load data

import cv2
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from LAB_01 import result
from LAB_08_NMS import IoU

annotations = pd.read_csv("numberplates/annotations.csv")
allnames = annotations.iloc[:,[0]].values
box_list = annotations.iloc[:,[3,4,5,6]].values
allnames = np.ndarray.flatten(allnames)
print(allnames)
print(box_list)

car_save_path = 'Used_Car/Plate/'
no_car_save_path = 'Used_Car/Not_plate/'
total_not_car = 0
total_car = 0

for i in range(len(allnames)):  # 50 times
    file = allnames[i]         # save file name
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    pic = cv2.imread('numberplates/'+file)
    pic_copy = pic.copy()
    ss.setBaseImage(pic)
    ss.switchToSelectiveSearchFast()
    results = ss.process()   # has 1000 of ROI
    car_count = 0
    no_car_count = 0
    total_counted = 0

    for found_box in results:     # for every image
        found_box_use = [found_box[0],found_box[1],found_box[0]+found_box[2],found_box[1]+found_box[3]]
        image_roi = pic_copy[found_box[1]:found_box[1]+found_box[3], found_box[0]:found_box[0]+found_box[2]]
        iou = IoU(found_box_use,box_list[i])
        if iou>0.7:
            if car_count < 16: # don't have enough memory for too many
                image_roi_use = cv2.resize(image_roi,(128,128))
                image_roi_use = image_roi_use.reshape((128,128,3))
                cv2.imwrite(car_save_path+'Plate'+str(total_car)+'.png',image_roi_use)
                total_car+=1
                car_count+=1

        if iou < 0.3:  # jaha nmbr plate k sath km roi hy
            if no_car_count < 16:
                image_roi_use = cv2.resize(image_roi, (128, 128))
                image_roi_use = image_roi_use.reshape((128, 128, 3))
                cv2.imwrite(no_car_save_path + 'Not_plate' + str(total_not_car) + '.png', image_roi_use)
                total_not_car += 1
                no_car_count += 1

        if total_counted > 999:
            break
        total_counted+=1


def load_data():
    data = load_files('use_car')
    filenames = data['filenames']
    X = []
    Y = []
    not_count = 0
    for name in filenames:
        if 'not' in name:
            not_count+=1
            if not_count < 2300:
                img = cv2.imread(name)
                X.append(img)
                Y.append(0)
            else:
                pass
        else:
            img = cv2.imread(name)
            X.append(img)
            Y.append(1)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y

