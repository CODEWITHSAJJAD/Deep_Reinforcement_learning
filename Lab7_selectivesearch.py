import cv2
import random
import numpy as np
from nltk.sentiment.util import output_markdown
image =cv2.imread("img.jpg")
image=cv2.resize(image,(512,512))
print("Calculating RoI using selective search")
ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()
rects=ss.process()
print(rects)
print("total regions generated =",len(rects))
for i in range(0,len(rects),10):
    output=image.copy()
    for (x,y,w,h) in rects[i:i+4030]:
        color=[random.randint(0,255) for j in range(0,3)]
        cv2.rectangle(output,(x,y),(x+w,y+h),color,2)
    cv2.imshow('birds',output)
    key=cv2.waitKey(0)
    if key==ord("Q"):
        break
cv2.destroyAllWindows()
