# rectangles on picture

import cv2
import random
import numpy as np

image = cv2.imread("Bird.png")
image = cv2.resize(image,(512,512))
print("Calculating ROI using selective search")

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)  # processing on ss
ss.switchToSelectiveSearchQuality()
rects = ss.process()  # rects for rectangles
print(rects)
print("total regions generated: ",len(rects))

for i in range(0,len(rects),100):  # runs rects time, every time it show 10 rects
    output = image.copy()
    for(x,y,w,h) in rects[i:i+100]:
        color = [random.randint(0,255) for j in range(0,3)]  # every time color of rectangle will change and loop means 3 channels
        cv2.rectangle(output,(x,y),(x+w,y+h), color,2) # (x,y) = top left pixel, (x+w,y+h)=bottom right pixel, 2= width of line from which triangle draw
    cv2.imshow('Birds',output)
    key = cv2.waitKey(0)
    if key==ord("q"):  # press q then no image will display
        break
cv2.destroyAllWindows() # all windows clear from memory

np.save("Rects",rects)
# save numpy array = np.save('my_array.npy', arr)
# load numpy array = loaded_arr = np.load('my_array.npy')
# website named as roboflow-> different datasets are there -> download any dataset in csv -> we apply yolo and rcn
# how to save and load  a numpy array -> use that numpy of rects as boxes in lab 8