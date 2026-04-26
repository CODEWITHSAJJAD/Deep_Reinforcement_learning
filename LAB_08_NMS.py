# NMS -> overlapped bounding boxes ko khtm kr k 1 bounding box bnata hy
# object detection : iou
import numpy as np
def NMS(boxes, overlapThresh):  # boxes are rectangles, overlapthresh = kitna overlaped ko khtm krna hy
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == 'i':  # i for integer  means if int then make float
        boxes = boxes.astype("float")
    pick = []
    # grab the coordinates of the bounding box
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2-x1+1) * (y2-y1+1)
    idxs = np.argsort(y2) # sort list of rects and give some indexes , last index contain rectangle with highest length
    # jis ki lenght bri hy us ko utha k us k according dekhna hy k overlaped kn kn sy ho rhy hain
    while len(idxs) > 0:
        last = len(idxs) -1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.maximum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2, - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

# in bounding boxes we have detail of 4 pixels

def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] +1) * (boxA[3] - boxA[1] +1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea)
    return iou