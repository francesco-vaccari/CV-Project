import utilsBenchmark as utils
import cv2 as cv

def MOG2_1(ct,log):
    cam = cv.VideoCapture("./video/out10.mp4")

    history = 500
    treshold = 16
    shadows = False

    b_subtractor = cv.createBackgroundSubtractorMOG2(history, treshold, shadows)

    treshold_accuracy = 245
    area_lower = 1000
    area_higher = 8000

    log.write("\nbenchmarking MOG2 with history "+str(history)+", treshold "+str(treshold)+" and shadows:"+str(shadows)
              +"\n treshold accuracy\t" + str(treshold_accuracy) + "\narea between " + str(area_lower) + " and " +
              str(area_higher))
    utils.subtractAndBound(cam, b_subtractor, treshold_accuracy, area_lower, area_higher, ct, log, "MOG2_1")

def MOG2_2(ct,log):
    cam = cv.VideoCapture("./video/out10.mp4")

    history = 1000
    treshold = 18
    shadows = False

    b_subtractor = cv.createBackgroundSubtractorMOG2(history, treshold, shadows)

    treshold_accuracy = 245
    area_lower = 1000
    area_higher = 8000

    log.write("\nbenchmarking MOG2 with history "+str(history)+", treshold "+str(treshold)+" and shadows:"+str(shadows)
              +"\n treshold accuracy\t" + str(treshold_accuracy) + "\narea between " + str(area_lower) + " and " +
              str(area_higher))
    utils.subtractAndBound(cam, b_subtractor, treshold_accuracy, area_lower, area_higher, ct, log, "MOG2_2")

def MOG2_3(ct,log):
    cam = cv.VideoCapture("./video/out10.mp4")

    history = 500
    treshold = 16
    shadows = True

    b_subtractor = cv.createBackgroundSubtractorMOG2(history, treshold, shadows)

    treshold_accuracy = 245
    area_lower = 1000
    area_higher = 8000

    log.write("\nbenchmarking MOG2 with history "+str(history)+", treshold "+str(treshold)+" and shadows:"+str(shadows)
              +"\n treshold accuracy\t" + str(treshold_accuracy) + "\narea between " + str(area_lower) + " and " +
              str(area_higher))
    utils.subtractAndBound(cam, b_subtractor, treshold_accuracy, area_lower, area_higher, ct, log, "MOG2_3")

