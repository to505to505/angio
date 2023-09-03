import os
import time
import random
import numpy
import cv2
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score #recall = sensitivity, precision = PPV
import skimage


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


"""METRICS"""
def metricsCalculator(y_true, y_pred):
    score_jaccard = jaccard_score(y_true, y_pred, pos_label=255)
    score_f1 = f1_score(y_true, y_pred, pos_label=255)
    score_recall = recall_score(y_true, y_pred, pos_label=255)
    score_precision = precision_score(y_true, y_pred, pos_label=255)
    score_acc = accuracy_score(y_true, y_pred)
    score_auc = roc_auc_score(y_true, y_pred)

    scores = [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_auc]

    print(f"\t=>Jaccard: {score_jaccard:1.4f} - F1: {score_f1:1.4f} - Recall: {score_recall:1.4f} - Precision: {score_precision:1.4f} - Acc: {score_acc:1.4f} - AUC: {score_auc:1.4f}\n")


    return scores

def skelEndpoints(maskArray):
    skel = skimage.morphology.skeletonize(maskArray.astype('bool'))
    skel = numpy.uint8(skel>0)

    # Apply the convolution.
    kernel = numpy.uint8([[1,  1, 1],
    [1, 10, 1],
    [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # Look through to find the value of 11.
    # This returns a mask of the endpoints, but if you
    # just want the coordinates, you could simply
    # return numpy.where(filtered==11)
    out = numpy.zeros_like(skel)
    out[numpy.where(filtered==11)] = 1
    # endCoords = numpy.where(filtered==11)
    # endCoords = list(zip(*endCoords))
    # startPoint = endCoords[0]
    # endPoint = endCoords[1]

    # print(f"Skel starts at {startPoint} and finishes at {endPoint}")

    # print(sum(out))

    out = out.astype('uint8')*255

    return out


def crudeMaskGenerator(maskArray):
    skel = skimage.morphology.skeletonize(maskArray.astype('bool'))
    skel = numpy.uint8(skel>0)
    radius = 15


    crudeMask = numpy.zeros_like(skel)
    skelPoints = numpy.argwhere(skel>0)

    # Create a circular mask to dilate the skel
    y, x = numpy.ogrid[-radius:radius+1,
                   -radius:radius+1]

    circleMask = x**2 + y**2 <= radius**2

    for i, point in enumerate(skelPoints[:-1]):
        yPos = point[0]
        xPos = point[1]

        if (yPos < skel.shape[0]-radius and xPos < skel.shape[1]-radius):
            if (yPos > radius and xPos > radius):

                crudeMask[int(yPos-radius):int(yPos+radius+1),
                        int(xPos-radius):int(xPos+radius+1)] += circleMask


    crudeMask = crudeMask>0

    return crudeMask.astype('uint8')*255
