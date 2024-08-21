import cv2
import numpy as np


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
        for i, box in enumerate(initial_boxes):
            self.tracker_instances.append(self.tracker.create())
            
            self.tracker_instances[i].init(initial_frame, box)
    
    def update(self, frame):
        results = []
        boxes = []

        for i, instance in enumerate(self.tracker_instances):
            success, box = instance.update(frame)
            if not success:
                self.reinitialize_tracker(frame, i)
            else:
                self.update_tracker(frame, i, box)
            results.append(success)
            boxes.append(box)
        
        return results, boxes
    
    def update_tracker(self, frame, i, box):
        # receives bounding box surrounding the object in the frame and the index of the tracker that found it

        # if the tracker is successful, we should update the tracker with the new frame and bounding box
        # to build a history of the color histograms or feature points
        # or any other information that can help finding again the object if the tracker loses it
        
        self.color_histogram_history[i] = [] # something like this
        self.feature_points_history[i] = [] # something like this

    def reinitialize_tracker(self, frame, i):
        # receives the frame and the index of the tracker that lost the object

        # if a tracker loses the object, we should look for it in the frame and reinitialize the tracker
        # we can use a color histogram and the average size of the bounding box previous to losing it to find the object
        # or we can use feature points to find the object (SIFT?)

        color_history = self.color_histogram_history[i] # something like this
        features_history = self.feature_points_history[i] # something like this



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