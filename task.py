import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from LAB_08_NMS import NMS

def rcnn(image, model):   # create for prediction
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # create selective object
    ss.setBaseImage(image)   # to generate region of interest
    ss.switchToSelectiveSearchFast()
    results = ss.process()
    copy = image.copy()  # region generate
    positive_boxes = []
    for box in results:  # some regions are of work
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h

        roi = image[y1:y2, x1:x2]  # pick some pixels of copied image
        roi = cv2.resize(roi, (128, 128))  # resize to model input size
        roi_use = roi.reshape((1, 128, 128, 3))  # reshape for model input
        class_pred = model.predict(roi_use)[0][0]  # model prediction

        if class_pred > 0.98:  # threshold for positive prediction
            positive_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 0), 5)

    cleaned_boxes = NMS(np.array(positive_boxes), 0.1)
    for clean_box in cleaned_boxes:
        clean_x1, clean_y1, clean_x2, clean_y2 = clean_box
        cv2.rectangle(copy, (clean_x1, clean_y1), (clean_x2, clean_y2), (0, 255, 0), 3)

    plt.imshow(copy)
    plt.axis('off')  # Hide axes
    plt.show()

def main(image_name, model_path):
    test_img = cv2.imread(image_name)
    model = tf.keras.models.load_model(model_path)  # Load the model
    rcnn(test_img, model)  # Run the R-CNN prediction

# Example usage
main('test_img1.jpg', 'Base_model.keras')
