import cv2 as cv

def subtract(videoname):
    cam = cv.VideoCapture(videoname)

    #fine-tuning values
    b_subtractor = cv.createBackgroundSubtractorMOG2(1000, 18, False)
    treshold_accuracy = 245
    area_lower = 1000
    area_higher = 8000

    while True:
        ret, frame = cam.read()

        foreground = b_subtractor.apply(frame)
        _, threshold = cv.threshold(foreground, treshold_accuracy, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv.contourArea(contour)
            if (area > area_lower)  and  (area < area_higher):
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        final = cv.resize(frame,(1200,750), 1, 1)
        cv.imshow('frame', final)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    subtract("./videos/out10.mp4")


