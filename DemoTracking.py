import cv2 as cv

# check out the problem at line 43, as that will be a major issue when implementing this
# once that is resolved, it should be possible to define 12 trackers to track each individual player and
# also implement a routine to "refresh" when one of the tracked players is lost for a long enough period of time.

#cam = cv.VideoCapture("./video/out10.mp4")
cam = cv.VideoCapture(0)

tracker = cv.TrackerCSRT.create()
b_subtractor = cv.createBackgroundSubtractorMOG2()
treshold_accuracy = 245
area_lower = 5000
area_higher = 8000

ret, frame = cam.read()
initBB = None

while True:
    ret, frame = cam.read()

    if initBB is not None:
        (success, box) = tracker.update(frame)

        if (success):
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    final = cv.resize(frame, (1200, 750), 1, 1)
    cv.imshow("Final", final)
    key = cv.waitKey(1) & 0xFF

    foreground = b_subtractor.apply(frame)
    if key == ord("s"):
        _, threshold = cv.threshold(foreground, treshold_accuracy, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours :
            area = cv.contourArea(contour)
            if (area > area_lower) and (area < area_higher):
                x, y, w, h = cv.boundingRect(contour)
                #initBB = [x,y,w,h]
        #somehow, the tracker won't initiate with the Rectangle produced by boundinhRect
        #need to find a way to fix it in order to make the whole procedure function
        #the next is assignment to test the rest of the tracking
        initBB = cv.selectROI("Final",final)

        tracker.init(frame, initBB)

    elif key == ord("q"):
        break

cam.release()
cv.destroyAllWindows()