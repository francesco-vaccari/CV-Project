import cv2
import numpy as np
import histogram as his
import Detection as det
import Annotation
import enum
import math

annotations_folder = 'annotations'
Status = enum.Enum('Status', ['ACTIVE', 'LOST', 'WAITING','OVERLAPPING', 'STATIC'])
WAIT_TIME = 10
MAX_OVERLAP_TIME = 30
MAX_OVERLAP_AREA = 0.5
MAX_STATIC_TIME = 50
MAX_STATIC_ERROR = 20
FRAMES_BEFORE_CHECKING = 10
BGSUB = det.BackgroundSubtractor(bg_path='background_image.jpg', threshold=50)

def boxArea(box):
    area = abs(box[2]-box[0])*abs(box[3]-box[1])
    return area

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
        self.frames_toSkip = FRAMES_BEFORE_CHECKING
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
        if self.frames_toSkip == 0:
            self.check_Static(boxes, results)
            self.check_Overlap(boxes, results)
        else:
            self.frames_toSkip -= 1
        
        return results, boxes
    
    def update_tracker(self, frame, i, box):
        # receives bounding box surrounding the object in the frame and the index of the tracker that found it

        # if the tracker is successful, we should update the tracker with the new frame and bounding box
        # to build a history of the color histograms or feature points
        # or any other information that can help finding again the object if the tracker loses it

        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)

        self.color_histogram_history[i].append(his.extract_histograms(frame, box))  # something like this
        self.box_size_history[i].append(boxArea(box))
        self.center_history[i].append(center)
        # self.feature_points_history[i] = [] # something like this

    def reinitialize_tracker(self, frame, i):
        # receives the frame and the index of the tracker that lost the object

        # if a tracker loses the object, we should look for it in the frame and reinitialize the tracker
        # we can use a color histogram and the average size of the bounding box previous to losing it to find the object
        # or we can use feature points to find the object (SIFT?)

        #color_history = self.color_histogram_history[i] # something like this
        #features_history = self.feature_points_history[i] # something like this

        # apply motion detection
        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        new_boxes = det.extract_boxes(bg)
        for j, box in enumerate(new_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        frame2 = frame
        frame2 = cv2.resize(frame2, (800, 800))
        cv2.imshow("Snap", frame2)

        min_diff = -1
        meansize = 0
        print("Last boxsize was" + str(self.box_size_history[i][-1]))
        for s, size in enumerate(self.box_size_history[i]):
            meansize += size
        meansize = meansize / s

        # THERE'S PROBABLY A PROBLEM HERE
        for b, box in enumerate(new_boxes) :
            boxsize = boxArea(box) # I would guess that boxsize calculations are
            # broken at some point
            #print(str(boxsize)+ " against "+ str(meansize))
            if ((meansize*0.5)<=boxsize<=(meansize*1.5)): # this is also a good candidate to be broken
                last_diff = his.compareToHistory(his.extract_histograms(frame, box), self.color_histogram_history[i])
                # I already checked Histograms comparison and it seems to work fine
                #print(str(last_diff)+" is the error of the histograms")
                if (last_diff < min_diff or min_diff == -1):
                    min_diff = last_diff # also this search for a minimum seems to be reliable
                    min_b = b
            #print(str(min_diff)+" is current minimum")

        if min_diff == -1 : # somehow we get inside this way too often, mainly cause the programs calculates areas to
            # be very small, even when they are pretty big
            self.status[i] = Status.WAITING
            self.waitouts[i] = WAIT_TIME
            return False, (0,0,0,0)
        else:
            print("Found Rect "+str(new_boxes[min_b])+" area is "+str(abs(new_boxes[min_b][2]-new_boxes[min_b][0])*abs(
                new_boxes[min_b][3]-new_boxes[min_b][1]))+"\n")
            self.tracker_instances[i].init(frame, new_boxes[min_b])

            self.status[i] = Status.ACTIVE
            return True, new_boxes[min_b]

    def check_Overlap(self, boxes, results):
        for i in range(0, len(boxes)):
            for j in range(i+1, len(boxes)):

                if results[i] and results[j]:

                    first = boxes[i]
                    second = boxes[j]

                    if (((first[0] <= second[0] <= first[0] + first[2]) and (first[1] <= second[1] <= first[1] +
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

                        x1 = max(first[0], second[0])
                        y1 = max(first[1], second[1])
                        x2 = min(first[2], second[2])
                        y2 = min(first[3], second[3])
                        third = (x1,x2,y1,y2)

                        area1 = boxArea(first)
                        area2 = boxArea(second)
                        area3 = boxArea(third)

                        if area3 == 0 :
                            #print("No Overlap here."+ str(i)+ " "+ str(j) + "\n")
                            break
                        elif area1 == 0 or area2 == 0:
                            #print("Somehow, an area was zero. "+ str(i)+ " "+ str(j) + "\n")
                            break

                        if (area3 / area1) >= MAX_OVERLAP_AREA or (area3 / area2) >= MAX_OVERLAP_AREA:
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
                        print("STATIC "+str(i)+" eliminated")
                        self.status[i] = Status.LOST
                        self.static_time[i] = 0
                    else:
                        self.static_time[i] += 1
                        self.status[i] = Status.STATIC
                else:
                    self.static_time = 0
                    self.status = Status.ACTIVE





class DenseOpticalFlowTracker:
    def __init__(self, tracker_name, frame, initial_boxes, points_sampling='gaussian25'):
        available_trackers = ["DISOpticalFlow", "FarnebackOpticalFlow"]
        if tracker_name not in available_trackers:
            raise Exception("Tracker not available. Available trackers are " + str(available_trackers))

        available_points_sampling = ['gaussian9', 'gaussian16', 'gaussian25', 'center', 'random9', 'random16', 'random25']
        if points_sampling not in available_points_sampling:
            raise Exception("Points sampling not available. Available points sampling are " + str(available_points_sampling))
        
        if tracker_name == "DISOpticalFlow":
            self.tracker = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST) # PRESET_ULTRAFAST, PRESET_FAST AND PRESET_MEDIUM
        elif tracker_name == "FarnebackOpticalFlow":
            self.tracker = cv2.FarnebackOpticalFlow.create()
        
        self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        self.boxes = []
        for box in initial_boxes:
            x, y, w, h = box
            self.boxes.append([x, y, x + w, y + h])

        self.points_sampling = points_sampling
        self.points = []
        for i in range(len(self.boxes)):
            self.points.append(self.sample_points(self.boxes[i]))
    
    def sample_points(self, box, single_point_resample=False):
        points = []
        box = box[0], box[1], box[2], box[3]
        min_x = min(box[0], box[2])
        max_x = max(box[0], box[2])
        min_y = min(box[1], box[3])
        max_y = max(box[1], box[3])
        box = [min_x+1, min_y+1, max_x-1, max_y-1]
        if self.points_sampling == 'center':
            point = [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
            points.append(point)
            if single_point_resample:
                return point
        elif self.points_sampling == 'random9':
            for i in range(9):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random16':
            for i in range(16):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'random25':
            for i in range(25):
                point = [np.random.randint(box[0], box[2]), np.random.randint(box[1], box[3])]
                points.append(point)
                if single_point_resample:
                    return point
        elif self.points_sampling == 'gaussian9' or self.points_sampling == 'gaussian16' or self.points_sampling == 'gaussian25':
            num_points = int(self.points_sampling[8:])
            mean_x = (box[0] + box[2]) / 2
            mean_y = (box[1] + box[3]) / 2
            std_x = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            std_y = (math.fabs(box[0] - box[2]) + math.fabs(box[1] - box[3])) / 6
            for i in range(num_points):
                sampled_x = np.random.normal(loc=mean_x, scale=std_x)
                sampled_y = np.random.normal(loc=mean_y, scale=std_y)
                sampled_x = np.clip(sampled_x, box[0], box[2])
                sampled_y = np.clip(sampled_y, box[1], box[3])
                points.append([int(sampled_x), int(sampled_y)])
                if single_point_resample:
                    return [int(sampled_x), int(sampled_y)]

        return points
    
    def update(self, frame):
        self.next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.next_frame_color = frame
        flow = self.tracker.calc(self.previous_frame, self.next_frame, None)

        result = self.update_boxes(flow)

        for i, res in enumerate(result):
            if not res:
                # find again the object and resample the points to track
                # if the box is found set result[i] to True
                # otherwise leave it to False
                pass
                

        self.previous_frame = self.next_frame

        boxes = []
        for box in self.boxes:
            x, y, x2, y2 = box
            boxes.append([x, y, x2 - x, y2 - y])
        
        return result, boxes

    def update_boxes(self, flow):
        success = [True for i in range(len(self.boxes))]

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        frame_copy = self.next_frame_color.copy()

        for i, points in enumerate(self.points):
            for j, point in enumerate(points):
                x, y = point
                magnitude_point = magnitude[y, x]
                angle_point = angle[y, x]
                delta_x = magnitude_point * math.cos(angle_point)
                delta_y = magnitude_point * math.sin(angle_point)
                new_x = x + delta_x
                new_y = y + delta_y
                new_x = np.clip(new_x, 0, self.next_frame.shape[1]-1)
                new_y = np.clip(new_y, 0, self.next_frame.shape[0]-1)
                self.points[i][j] = [int(new_x), int(new_y)]
            
            x_min = self.next_frame.shape[1] + 1
            y_min = self.next_frame.shape[0] + 1
            x_max = -1
            y_max = -1
            for j, point in enumerate(self.points[i]):
                cv2.circle(frame_copy, (point[0], point[1]), 4, (0, 255, 0), -1)
                x, y = point
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            cv2.circle(frame_copy, (int(center[0]), int(center[1])), 8, (255, 0, 0), -1)

            bg = BGSUB.apply(self.next_frame_color)
            bg = det.preprocess(bg)
            new_boxes = det.extract_boxes(bg)

            for box in new_boxes:
                x, y, x2, y2 = box
                if center[0] >= x and center[0] <= x2 and center[1] >= y and center[1] <= y2:
                    # maybe instead of just substituting the box
                    # I can take move the box of before with the averaged flow of all points
                    # and then increase or decrease the coordinates of the edges according to the size of the box found with detector
                    self.boxes[i] = box
                    cv2.rectangle(frame_copy, (x, y), (x2, y2), (255, 0, 0), 4)
                    success[i] = True
                    break
            
            # resample the points that are not inside the bounding box
            # maybe if more than a percent of points are outside the declare tracking lost
            for j, point in enumerate(self.points[i]):
                x1 = self.boxes[i][0]
                y1 = self.boxes[i][1]
                x2 = self.boxes[i][2]
                y2 = self.boxes[i][3]
                if point[0] < x1 or point[0] > x2 or point[1] < y1 or point[1] > y2:
                    self.points[i][j] = self.sample_points(self.boxes[i], single_point_resample=True)

        cv2.imshow('tracker', frame_copy)
        return success



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