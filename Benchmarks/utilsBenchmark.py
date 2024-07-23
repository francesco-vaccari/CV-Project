import cv2 as cv
import datetime

def initialize_timestamp(name):
    ct = datetime.datetime.now()
    log = open(name+"_log.txt", "w")

    return ct, log

def cleanup(ct, log):
    log.write("\nProcess ended, closing at: "+str(ct))
    log.close()


def subtractAndBound(cam, b_subtractor, treshold_accuracy, area_lower, area_higher, ct, log, name):
    bounding_count = 0
    current_frame = 0

    #video output, to be fixed. Remember the other 2 commented lines of code regarding video output.
    #out_vid = cv.VideoWriter(name+"_bench.avi",cv.VideoWriter_fourcc(*"MJPG"), cam.get(cv.CAP_PROP_FPS), (int(cam.get(3)), int(cam.get(4))))

    log.write("\nstarting at "+str(ct))
    while True:
        ret, frame = cam.read()

        foreground = b_subtractor.apply(frame)
        _, threshold = cv.threshold(foreground, treshold_accuracy, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        current_count = 0
        current_frame = current_frame + 1
        for contour in contours:
            area = cv.contourArea(contour)
            if (area > area_lower) and (area < area_higher):
                if current_frame > 200:
                    current_count = current_count + 1
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        bounding_count = bounding_count + current_count
        final = cv.resize(frame, (1200, 750), 1, 1)
        #out_vid.write(final)
        #cv.imshow('frame', final)
        print("\nFrame "+str(current_frame)+" just finished")
        # final2 = cv.resize(threshold, (1000, 750), 0.5, 0,5)
        # cv.imshow('foreground', final2)

        if (cv.waitKey(1) & 0xFF == ord('q')) or (current_frame == 500):
            bounding_count = bounding_count / current_frame
            log.write("\nMean number of bounding boxes per frame:\t")
            log.write(str(bounding_count))
            break

    log.write("\nEnded at "+str(ct))
    cam.release()
    #out_vid.release()
    cv.destroyAllWindows()

