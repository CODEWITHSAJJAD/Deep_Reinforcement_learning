import numpy as np
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind =="i":
        boxes = boxes.astype('float')
    pick = []
    # Grab boundry coordinate
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 +1) * (y2 - y1 +1)
    indexes = np.argsort(y2)

    while len(indexes)>0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.maximum(x2[i], x2[indexes[:last]])
        yy2 = np.maximum(y2[i], y2[indexes[:last]])
        w = np.maximum(0,xx2 - xx1 +1)
        h = np.maximum(0,yy2 - yy1 +1)
        overlap = (w*h)/area[indexes[:last]]
        indexes = np.delete(indexes, np.concatenate(([last],np.where(overlap>overlapThresh)[0])))
#save np array and load it here, show it in boxes


def IOU(boxA,boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1)* max(0, yB-yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1]+ 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1]+ 1)
    iou = interArea/float(boxAArea + boxBArea - interArea)
    return iou