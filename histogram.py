import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

colors = ('b', 'g', 'r')
COMP_METHOD = cv.HISTCMP_CORREL
def extract_histograms(frame, box):
    x, y, h, w = box

    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y:(y + w), x:x + h] = 255
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)

    histograms = []

    for i, col in enumerate(colors):
        histograms.append(cv.calcHist([frame], [i], mask, [256], [0, 256]))

    return histograms

def compareToHistory (toCompare, history):
    diff = []

    for c, col in enumerate(colors):
        cv.normalize(toCompare[c], toCompare[c], alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    for i, histo in enumerate(history):
        for c, col in enumerate(colors):
            cv.normalize(history[i][c], history[i][c], alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            diff.append(cv.compareHist(toCompare[c], history[i][c], COMP_METHOD))

    error = 0
    for i, dif in enumerate(diff) :
        error += dif
    error = error / len(diff)
    #print(error)

    return error

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