import pickle

from skimage.transform import resize
import numpy as np
import cv2

MODEL = pickle.load(open("model.p", "rb"))

EMPTY= True
NOT_EMPTY = False


def get_parking_spots_bboxes(connected_components):
    (totalLabels,labbel_ids, values, controid) = connected_components

    slots = []
    coef = 1

    for i in range(1, totalLabels):
        # extract the cordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def empty_or_not(spot_bgr):
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY