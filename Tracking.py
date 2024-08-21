import cv2
import numpy as np
import histogram as his
import Detection as det
import Annotation
import enum

annotations_folder = 'annotations'
Status = enum.Enum('Status', ['ACTIVE', 'LOST', 'WAITING','OVERLAPPING', 'STATIC'])
WAIT_TIME = 10
MAX_OVERLAP_TIME = 30
MAX_OVERLAP_AREA = 0.5
MAX_STATIC_TIME = 30
MAX_STATIC_ERROR = 10
BGSUB = det.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)

class OpenCVTracker:
    def __init__(self, tracker_name, initial_frame, initial_boxes):
        available_trackers = ["CSRT", "MIL", "KCF", "DaSiamRPN", "GOTURN", "Nano", "Vit"]
        if tracker_name not in available_trackers:
            raise Exception("Tracker not available. Available trackers are " + str(available_trackers))

        if tracker_name == "CSRT":
            self.tracker = cv2.TrackerCSRT
        elif tracker_name == "MIL":
            self.racker = cv2.TrackerMIL
        elif tracker_name == "KCF":
            self.tracker = cv2.TrackerKCF
        elif tracker_name == "DaSiamRPN":
            self.tracker = cv2.TrackerDaSiamRPN
        elif tracker_name == "GOTURN":
            self.tracker = cv2.TrackerGOTURN
        elif tracker_name == "Nano":
            self.tracker = cv2.TrackerNano
        elif tracker_name == "Vit":
            self.tracker = cv2.TrackerVit

        self.tracker_instances = []
        self.color_histogram_history = []
        self.box_size_history = []
        self.waitouts = []
        self.status = []
        self.static_time =[]
        self.center_history = []
        for i, box in enumerate(initial_boxes):
            self.tracker_instances.append(self.tracker.create())
            
            self.tracker_instances[i].init(initial_frame, box)

            self.status.append(Status.ACTIVE)

            self.color_histogram_history.append([])
            self.box_size_history.append([])
            self.waitouts.append(0)
            self.static_time.append(0)
            self.center_history.append([])
    
    def update(self, frame):
        results = []
        boxes = []

        for i, instance in enumerate(self.tracker_instances):
            if self.status[i] == Status.WAITING:
                if self.waitouts[i] == 0:
                    success, box = self.reinitialize_tracker(frame, i)
                    print(
                        "Attempt after WAIT, result " + str(self.status[i]) + " from " + str(i) + " waitout is " + str(
                            self.waitouts[i]) + " \n")
                    if success:
                        self.update_tracker(frame,i,box)
                else:
                    self.waitouts[i] -= 1
                    success = False
                    box = (0,0,0,0)
            elif self.status[i] == Status.LOST or self.status[i] == Status.OVERLAPPING:
                success, box = self.reinitialize_tracker(frame, i)
                print(
                    "Attempt after CHECKS, result " + str(self.status[i]) + " from " + str(i) + " \n")
                if success:
                    self.update_tracker(frame, i, box)
            else:
                success, box = instance.update(frame)
                if not success:
                    self.status[i] = Status.LOST
                    success, box = self.reinitialize_tracker(frame, i)
                    print("Immediate Attempt, result " + str(self.status[i]) + " from " + str(i) + "\n")
                if success:
                    self.update_tracker(frame, i, box)
            results.append(success)
            boxes.append(box)

        self.check_Static(boxes, results)
        self.check_Overlap(boxes, results)
        
        return results, boxes
    
    def update_tracker(self, frame, i, box):
        # receives bounding box surrounding the object in the frame and the index of the tracker that found it

        # if the tracker is successful, we should update the tracker with the new frame and bounding box
        # to build a history of the color histograms or feature points
        # or any other information that can help finding again the object if the tracker loses it

        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)

        self.color_histogram_history[i].append(his.extract_histograms(frame, box))  # something like this
        self.box_size_history[i].append(abs(box[0]-box[2])*abs(box[1]-box[3]))
        self.center_history[i].append(center)
        # self.feature_points_history[i] = [] # something like this

    def reinitialize_tracker(self, frame, i):
        # receives the frame and the index of the tracker that lost the object

        # if a tracker loses the object, we should look for it in the frame and reinitialize the tracker
        # we can use a color histogram and the average size of the bounding box previous to losing it to find the object
        # or we can use feature points to find the object (SIFT?)

        #color_history = self.color_histogram_history[i] # something like this
        #features_history = self.feature_points_history[i] # something like this

        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        new_boxes = det.extract_boxes(bg)

        min_diff = -1
        meansize = 0
        for s, size in enumerate(self.box_size_history[i]):
            meansize += size
        meansize = meansize / len(self.box_size_history[i])

        for b, box in enumerate(new_boxes) :
            last_diff = his.compareToHistory(his.extract_histograms(frame, box), self.color_histogram_history[i])
            boxsize = abs(box[0]-box[2])*abs(box[1]-box[3])
            if (last_diff < min_diff or min_diff == -1) and ((meansize*0.5)<=boxsize<=(meansize*2)):
                min_diff = last_diff
                min_b = b

        if min_diff == -1 :
            self.status[i] = Status.WAITING
            self.waitouts[i] = WAIT_TIME
            return False, (0,0,0,0)
        else:
            print("Found Rect "+str(new_boxes[min_b])+" area is "+str(new_boxes[min_b][2]*new_boxes[min_b][3])+"\n")
            self.tracker_instances[i].init(frame, new_boxes[min_b])

            self.status[i] = Status.ACTIVE
            return True, new_boxes[min_b]

    def check_Overlap(self, boxes, results):
        for i in range(0, len(boxes)):
            for j in range(i+1, len(boxes)):

                if results[i] and results[j]:

                    first = boxes[i]
                    second = boxes[j]

                    if not(((first[0] <= second[0] <= first[0] + first[2]) and (first[1] <= second[1] <= first[1] +
                         first[3])) or ((first[0] <= second[0] <=first[0] + first[2]) and (first[1] <= second[1] +
                         second[3] <= first[1] + first[3])) or ((first[0] <= second[0] + second[2] <= first[0] +
                         first[2])  and (first[1] <= second[1] <= first[1] + first[3])) or ((first[0] <= second[0] +
                         second[2] <= first[0] + first[2]) and (first[1] <= second[1] + second[3]<= first[1] + first[3]))
                         or ((second[0] <= first[0] <= second[0] + second[2]) and (second[1] <= first[1] <= second[1] +
                         second[3])) or ((second[0] <= first[0] <=second[0] + second[2]) and (second[1] <= first[1] +
                         first[3] <= second[1] + second[3])) or ((second[0] <= first[0] + first[2] <= second[0] +
                         second[2])  and (second[1] <= first[1] <= second[1] + second[3])) or ((second[0] <= first[0] +
                         first[2] <= second[0] + second[2]) and (second[1] <= first[1] + first[3]<= second[1] + second[3]))
                         ):
                        break

                    x1 = max(first[0], second[0])
                    y1 = max(first[1], second[1])
                    x2 = min(first[2], second[2])
                    y2 = min(first[3], second[3])

                    area1 = abs(first[2] - first[0]) * abs(first[3] - first[1])
                    area2 = abs(second[2] - second[0]) * abs(second[3] - second[1])
                    area3 = abs(x2 - x1) * abs(y2 - y1)

                    if area3 == 0 :
                        #print("No Overlap here."+ str(i)+ " "+ str(j) + "\n")
                        break
                    elif area1 == 0 or area2 == 0:
                        #print("Somehow, an area was zero. "+ str(i)+ " "+ str(j) + "\n")
                        break

                    if area3 / area1 >= MAX_OVERLAP_AREA or area3 / area2 >= MAX_OVERLAP_AREA:
                        if area1 < area2:
                            self.status[i] = Status.OVERLAPPING
                            print(str(i)+ " Overlaps on "+str(j)+"\n")
                        else:
                            self.status[j] = Status.OVERLAPPING
                            print(str(j) + " Overlaps on " + str(i) + "\n")
                    elif self.status[i]==Status.OVERLAPPING or self.status[j] == Status.OVERLAPPING:
                        self.status[i] = Status.ACTIVE
                        self.status[j] = Status.ACTIVE

    def check_Static(self, boxes, results):
        for i, box in enumerate(boxes):
            if results[i]:
                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                old_center = self.center_history[i][-1]
                if abs(old_center[0] - center[0]) <= MAX_STATIC_ERROR and abs(
                        old_center[1] - center[1]) <= MAX_STATIC_ERROR:

                    if self.static_time == MAX_STATIC_TIME:
                        self.status[i] = Status.LOST
                        self.static_time[i] = 0
                    else:
                        self.static_time[i] += 1
                        self.status[i] = Status.STATIC
                else:
                    self.static_time = 0
                    self.status = Status.ACTIVE





class DenseOpticalFlowTracker:
    def __init__(self, tracker_name, frame, initial_boxes):
        available_trackers = ["DISOpticalFlow", "FarnebackOpticalFlow"]
        if tracker_name not in available_trackers:
            raise Exception("Tracker not available. Available trackers are " + str(available_trackers))
        
        if tracker_name == "DISOpticalFlow":
            self.tracker = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST) # PRESET_ULTRAFAST, PRESET_FAST AND PRESET_MEDIUM
        elif tracker_name == "FarnebackOpticalFlow":
            self.tracker = cv2.FarnebackOpticalFlow.create()
        
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.boxes = initial_boxes
    
    def update(self, frame):
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = self.tracker.calc(self.previous_frame, self.next_frame, None)

        result = self.update_boxes(flow)

        self.previous_frame = self.next_frame

        return result, self.boxes

    def update_boxes(self, flow):
        # apply the movement described by the optical flow found to the center of the boxes
        # or to a series of poinst within the box so that we can compute the average movement of the object and maybe also exclude outliers
        # that have values too far from the average


        # then use the mask to find the bounding box surrounding the center of the box predicted by the optical flow
        # or around the points moved with the flow
        # or use a motion detector to find the bounding box surrounding the object
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mask = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mask = np.where(mask > 50, 255, 0)
        mask = mask.astype(np.uint8)
        

        # update the boxes with the new bounding boxes found

        
        # if the bounding box is not found or if the flow for that box is 0 (means lost tracking)
        # or the flow of the different is chaotic
        # then we need to find the object in the frame and set again the box to the new found bounding box
        # we can use the same thing of the opencv trackers class


        # return whether the boxes were found or not
        return [True] * len(self.boxes)




class PyrLKOpticalFlowTracker:
    def __init__(self, frame, initial_boxes):
        self.tracker = cv2.SparsePyrLKOpticalFlow.create()
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.boxes = initial_boxes

        # we can use the center of the bounding boxes as the points to track like in the following line
        self.previous_points = np.array([[int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)] for box in initial_boxes], dtype=np.float32)
        # or we can maybe use a set of points within the bounding box

    def update(self, frame):
        # use current frame to compute optical flow already applied to the points given as input
        # success is a vector that indicates whether the flow was found or not for the specific point
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, success, error = self.tracker.calc(self.previous_frame, self.next_frame, self.previous_points, None)

        result = self.update_boxes(next_points, success, error)

        self.previous_frame = self.next_frame
        self.previous_points = next_points

        return result, self.boxes

    def update_boxes(self, next_points, success, error):
        # since the flow is not given (already applied to points) we cannot use to exclude bad points
        # the error is a vector that contains the error for the corresponding point (it is an error measure, don't know the meaning)
        # maybe we can use that to evaluate the quality of points found

        # using points found with the optical flow calculated, we need to update the bounding boxes
        # if we use multiple points for each bounding box then we need some variable to tell us which points belong to which box
        # and then we need to find the bounding box surrounding the points found (or the center point) and to do so we can use a motion detector

        # if the success for the center point (or the points) is 0 then it means that it was not possible to find an optical flow for that point
        # and so we need fo find again the object and reinitialize the previous points and the bounding box
        # to do so we can use the same methods as before
        
        # return a list of booleans indicating whether the box for the object tracked was found or not
        return True

class KalmanFilterTracker:
    def __init__(self, frame, initial_boxes):
        pass