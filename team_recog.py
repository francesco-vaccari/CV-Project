import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import histogram as his
import Detection as det
import Annotation as ann
from scipy import stats as stat

video = 'videos/refined2_short.mp4'
annotations_folder = 'annotations'
colors = ('b', 'g', 'r')

criteria = (cv.TERM_CRITERIA_EPS, 10, 1)
flags = cv.KMEANS_RANDOM_CENTERS

detector = det.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)

cam = cv.VideoCapture(video)
annotations = ann.Annotation(annotations_folder)



cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Frame',600,900)
cv.namedWindow('mask', cv.WINDOW_NORMAL)
cv.resizeWindow('mask',600,900)
cv.namedWindow("K-means clustering", cv.WINDOW_NORMAL)
cv.resizeWindow("K-means clustering",600,900)

n_frame = 0
mode = "k-means"

avg_compactness = 0
n_frames_TeamsHaveSameLabel = 0
correctly_labeled = 0
wrongly_labeled = 0
correctly_labeled_withTeamsSameLabel = 0
wrongly_labeled_withTeamsSameLabel = 0
correctly_labeled_onlyTeamsSameLabel = 0
wrongly_labeled_onlyTeamsSameLabel = 0

log_folder = "notes/"
log_name = "teamDetect_noSanitize_bgsubThresh50_noPreprocess"
log = open(log_folder+log_name+".txt", "w")

while True:
    ret, frame = cam.read()
    n_frame += 1

    cv.imshow('Frame', frame)

    mask = detector.apply(frame)
    masked_frame = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("mask", masked_frame)

    boxes = annotations.get_annotation(n_frame)

    selected_histr = []
    selected_features = []

    if mode == "k-means":
        for i, box in enumerate(boxes):
            x, y, h, w = box.x1, box.y1, box.x2, box.y2

            histr = his.get_histogram(masked_frame, box)
            histr = his.sanatise_histogram(histr)

            feature = his.histogram_to_features(histr)
            selected_histr.append(histr)
            selected_features.append(feature)

        toPass = np.array(selected_features)
        toPass = np.float32(toPass)
        compactness, labels, centers = cv.kmeans(toPass, 2, None, criteria, 10, flags)

        team1 = toPass[labels.ravel() == 0]
        team2 = toPass[labels.ravel() == 1]

        #print(compactness)
        avg_compactness += compactness
        identical_labels = False

        label_1, count1 = stat.mode(labels[0:6])
        label_2, count2 = stat.mode(labels[6:])

        if (label_1 == label_2):
            if count1 != 3 and count2 != 3:
                # print("The two teams have identical mode labels")
                n_frames_TeamsHaveSameLabel += 1
                identical_labels = True
            else:
                if count1 == 3:
                    if label_1 == 0:
                        label_1 = 1
                    else :
                        label_1 = 0
                elif count2 == 3:
                    if label_2 == 0:
                        label_2 = 1
                    else :
                        label_2 = 0


        pos = 0
        neg = 0
        for i, label in enumerate(labels):
            if i <= 5:
                if label == label_1:
                    pos += 1
                    correctly_labeled_withTeamsSameLabel += 1
                    if identical_labels:
                        correctly_labeled_onlyTeamsSameLabel += 1
                    else :
                        correctly_labeled += 1
                else:
                    neg += 1
                    wrongly_labeled_withTeamsSameLabel += 1
                    if identical_labels:
                        wrongly_labeled_onlyTeamsSameLabel += 1
                    else :
                        wrongly_labeled += 1
            else:
                if label == label_2:
                    pos += 1
                    correctly_labeled_withTeamsSameLabel += 1
                    if identical_labels:
                        correctly_labeled_onlyTeamsSameLabel += 1
                    else :
                        correctly_labeled += 1
                else:
                    neg += 1
                    wrongly_labeled_withTeamsSameLabel += 1
                    if identical_labels:
                        wrongly_labeled_onlyTeamsSameLabel += 1
                    else :
                        wrongly_labeled += 1

        #print("Number of assigned labels that correspond to the mode of the team " + str(pos))
        #print("Number of assigned labels that DON'T correspond to the mode of the team " + str(neg))

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            if label_1 == label_2 and labels[i] == label_1:
                cv.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 0), 4)
            elif labels[i] == label_1 and i<6:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            elif labels[i] == label_2 and i>=6:
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            else:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv.imshow("K-means clustering", frame)

    key = cv.waitKey(1) & 0xFF

    if key == ord("q") or n_frame == 1810:
        if(n_frame!=0):
            avg_compactness = avg_compactness / n_frame
            correctly_labeled_withTeamsSameLabel = correctly_labeled_withTeamsSameLabel / n_frame
            wrongly_labeled_withTeamsSameLabel = wrongly_labeled_withTeamsSameLabel / n_frame
        if(n_frames_TeamsHaveSameLabel !=0):
            correctly_labeled_onlyTeamsSameLabel = correctly_labeled_onlyTeamsSameLabel / n_frames_TeamsHaveSameLabel
            wrongly_labeled_onlyTeamsSameLabel = wrongly_labeled_onlyTeamsSameLabel / n_frames_TeamsHaveSameLabel
        n_frames_noSameLabels = n_frame - n_frames_TeamsHaveSameLabel
        if(n_frames_noSameLabels!=0):
            correctly_labeled = correctly_labeled / n_frames_noSameLabels
            wrongly_labeled = wrongly_labeled / n_frames_noSameLabels
        log.write("Average compactness of clusters:"+str(avg_compactness)+"\n")
        log.write("----------------------------------------------------------------\n")
        log.write("Total number of frames analysed: "+str(n_frame)+"\n")
        log.write("Average number of correctly assigned labels :" + str(correctly_labeled_withTeamsSameLabel)+"\n")
        log.write("Average number of wrongly assigned labels :" + str(wrongly_labeled_withTeamsSameLabel)+"\n")
        log.write("----------------------------------------------------------------\n")
        log.write("Numer of Frames where the teams have the same mode labels :" + str(n_frames_TeamsHaveSameLabel)+"\n")
        log.write("Average number of correctly assigned labels :" + str(correctly_labeled_onlyTeamsSameLabel)+"\n")
        log.write("Average number of wrongly assigned labels :" + str(wrongly_labeled_onlyTeamsSameLabel)+"\n")
        log.write("----------------------------------------------------------------\n")
        log.write("Number of Frames where the teams have different mode labels : "+ str(n_frames_noSameLabels)+"\n")
        log.write("Average number of correctly assigned labels :" + str(correctly_labeled)+"\n")
        log.write("Average number of wrongly assigned labels :" + str(wrongly_labeled)+"\n")

        break

    elif key == ord("s"):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(team1[:, 0], team1[:, 1], team1[:,2])
        ax.scatter(team2[:, 0], team2[:, 1], team2[:,2], c='r')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:,2], s=80, c='y', marker='s')

        ax.set_xlabel('Blue')
        ax.set_ylabel('Green')
        ax.set_zlabel('Red')

        plt.show()

    elif key ==ord("p"):
        box = cv.selectROI("Frame", frame)

        histr = his.get_histogram_alt(masked_frame, box)
        for i, col in enumerate(colors):
            plt.plot(histr[i], color=col)
            plt.xlim([0, 256])

        plt.show()

        histr = his.sanatise_histogram(histr, 0.5)
        for i, col in enumerate(colors):
            plt.plot(histr[i], color=col)
            plt.xlim([0, 256])

        plt.show()



log.close()
cam.release()
cv.destroyAllWindows()