import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

colors = ('b', 'g', 'r')

def extract_histograms(frame, box):
    x, y, h, w = box

    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y:(y + w), x:x + h] = 255
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)

    histograms = []

    for i, col in enumerate(colors):
        histograms.append(cv.calcHist([frame], [i], mask, [256], [0, 256]))

    return histograms



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