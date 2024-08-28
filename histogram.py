import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

colors = ('b', 'g', 'r')
COMP_METHOD = cv.HISTCMP_CORREL
def get_histogram(frame, box):
    x, y, h, w = box.x1, box.y1, box.x2, box.y2

    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y:(y + w), x:x + h] = 255

    histograms = []

    for i, col in enumerate(colors):
        toAppend = cv.calcHist([frame], [i], mask, [256], [0, 256])
        cv.normalize(toAppend, toAppend, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        histograms.append(toAppend)

    return histograms

def get_histogram_alt(frame, box):
    x, y, h, w = box

    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y:(y + w), x:x + h] = 255

    histograms = []

    for i, col in enumerate(colors):
        toAppend = cv.calcHist([frame], [i], mask, [256], [0, 256])
        cv.normalize(toAppend, toAppend, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        histograms.append(toAppend)

    return histograms

def compare_histogram_to_history(toCompare, history):
    if len(history) == 0:
        return 0
    
    diff = []

    for i, histo in enumerate(history):
        for c, col in enumerate(colors):
            diff.append(cv.compareHist(toCompare[c], history[i][c], COMP_METHOD))

    error = 0
    for i, dif in enumerate(diff) :
        error += dif
    error = error / len(diff)

    return error

def histogram_to_features(histograms):
    features = []

    for i, col in enumerate(colors):
        values = []
        #cv.normalize(histograms[i], histograms[i], alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        for j,val in enumerate(histograms[i]):
            values.append(val)
        features.append(values)

    toSend = np.vstack((np.array(features[0]), np.array(features[1]), np.array(features[2])))
    toSend = np.float32(toSend)

    return toSend

def sanatise_histogram(histogram):
    for c, col in enumerate(colors):
        histogram[c][0] = 0

        cv.normalize(histogram[c], histogram[c], alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return histogram

#----------------------------------------------------------------------------------------------------------------------
# here starts unimportant, testing stuff
def test_histogram_extraction():
    cam = cv.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        cv.imshow('Frame', frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("s"):
            box = cv.selectROI("Frame", frame)

            histr = extract_histograms(frame, box)

            for i, col in enumerate(colors):
                plt.plot(histr[i], color=col)
                plt.xlim([0, 256])

            plt.show()


        elif key == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()

#calls to test functions.
#test_histogram_extraction()