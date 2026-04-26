import cv2
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
from Lab8_NMS import IOU

annotations = pd.read_csv('numberplates/annotations.csv')
allnames = annotations.iloc[:,[0]].values
box_list = annotations.iloc[:,[3,4,5,6]].values
allnames = np.ndarray.flatten(allnames)
print(allnames)
print(box_list)

car_save_path = 'use_car/plate/'
no_car_save_path = 'use_car/not_plate/'
total_not_car = 0
total_car = 0
for i in range(len(allnames)):
    file=allnames[i]
    ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    pic=cv2.imread('numberplates/'+file)
    pic_copy=pic.copy()
    ss.setBaseImage(pic)
    ss.switchToSelectiveSearchFast()
    results=ss.process()
    car_count=0
    no_car_count=0
    total_counted=0
    for found_box in results:
        found_box_use=[found_box[0],found_box[1],found_box[0]+found_box[2],found_box[1]+found_box[3]]
        image_roi=pic_copy[found_box[1]:found_box[3]+found_box[1],found_box[0]:found_box[0]+found_box[2]]
        iou=IOU(found_box_use,box_list[i])
        if iou>0.7:
            if car_count<16:
                image_roi_use=cv2.resize(image_roi,(128,128))
                image_roi_use=image_roi_use.reshape(128,128,3)
                cv2.imwrite(car_save_path+'plate_'+str(total_car)+'.png',image_roi_use)
                total_car+=1
                car_count+=1
        if iou<0.3:
            if car_count<16:
                image_roi_use=cv2.resize(image_roi,(128,128))
                image_roi_use=image_roi_use.reshape(128,128,3)
                cv2.imwrite(no_car_save_path+'not_plate_'+str(total_not_car)+'.png',image_roi_use)
                total_not_car+=1
                no_car_count+=1
        if total_counted>999:
            break
import pandas as pd

def load_data():
    data = pd.read_csv('use_car/')
    filenames=data['filenames']
    x=[]
    y=[]
    not_count=0
    for name in filenames:
        if 'not' in name:
            not_count+=1
            if not_count <2300:
                img=cv2.imread(name)
                x.append(img)
                y.append(0)
            else:
                pass
        else:
            img=cv2.imread(name)
            x.append(img)
            y.append(1)
    x=np.asarray(x)
    y=np.asarray(y)
    return x,y


