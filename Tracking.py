import cv2
import numpy as np
import histogram as his
import Detection as det
import Annotation
import enum
import math

annotations_folder = 'annotations'
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

        self.history_length = 48
        self.tracker_histogram_history = []
        self.tracker_position_history = []
        self.tracker_box_size_history = []

        self.tracker_ready = [0]*len(initial_boxes)
        self.tracker_instances = []
        for i, box in enumerate(initial_boxes):
            self.tracker_instances.append(self.tracker.create())
            self.tracker_instances[i].init(initial_frame, box)
            self.tracker_histogram_history.append([])
            self.tracker_position_history.append([])
            self.tracker_box_size_history.append([])
        for i, box in enumerate(initial_boxes):
            self.update_history(i, box, initial_frame)
    
    def update(self, frame):
        results = []
        boxes = []

        for i, tracker_instance in enumerate(self.tracker_instances):
            if self.tracker_ready[i] == 0:
                success, box = tracker_instance.update(frame)
                results.append(success)
                boxes.append(box)
            else:
                results.append(False)
                boxes.append((0,0,0,0))
        
        for i, res in enumerate(results):
            if res:
                if self.is_tracking_lost(boxes[i], frame):
                    self.tracker_ready[i] += 1
                    results[i] = False
                else:
                    self.update_history(i, boxes[i], frame)
            if not res:
                bg = BGSUB.apply(frame)
                bg = det.preprocess(bg)
                motion_boxes = det.extract_boxes(bg)
                result, box = self.match_boxes(i, motion_boxes, frame)
                print('Matching box found for tracker ' + str(i) + ': ' + str(box))
                if result:
                    self.tracker_instances[i].init(frame, box)
                    self.tracker_ready[i] = 0
                    results[i] = True
                    boxes[i] = box
                    frame_copy  = frame.copy()
                    cv2.rectangle(frame_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 4)
                    cv2.imshow('tracker', frame_copy)
        
        results, boxes = self.handle_overlapping(results, boxes)
        
        return results, boxes

    def is_tracking_lost(self, box, frame): # check if the bounding box tracked contains some movement
        bg = BGSUB.apply(frame)
        bg = det.preprocess(bg)
        motion_boxes = det.extract_boxes(bg)

        if len(motion_boxes) == 0:
            return True

        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        for motion_box in motion_boxes:
            x1_m, y1_m, x2_m, y2_m = motion_box
            # if any corner is inside the other box then there is movement
            if x1_m <= x1 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y1 <= y2_m:
                return False
            if x1_m <= x1 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1_m <= x2 <= x2_m and y1_m <= y2 <= y2_m:
                return False
            if x1 <= x1_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y1_m <= y2:
                return False
            if x1 <= x1_m <= x2 and y1 <= y2_m <= y2:
                return False
            if x1 <= x2_m <= x2 and y1 <= y2_m <= y2:
                return False
        
        return True

    def update_history(self, tracker_index, box, frame): # update history for tracker tracker_index with color, position and size
        if len(self.tracker_histogram_history[tracker_index]) == self.history_length:
            self.tracker_histogram_history[tracker_index].pop(0)
            self.tracker_position_history[tracker_index].pop(0)
            self.tracker_box_size_history[tracker_index].pop(0)
        self.tracker_histogram_history[tracker_index].append(his.get_histogram(frame, box))
        self.tracker_position_history[tracker_index].append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.tracker_box_size_history[tracker_index].append([box[2], box[3]])

    def match_boxes(self, tracker_index, matching_boxes, frame):
        hist_factor = 0.7
        pos_factor = 0.2
        size_factor = 0.1

        match_score = [0]*len(matching_boxes)
        frames_passed = self.tracker_ready[tracker_index] # frames passed since the tracking has been lost
        for i, box in enumerate(matching_boxes):
            box_histogram = his.get_histogram(frame, box)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            box_center = [x + w / 2, y + h / 2]

            histogram_comparison = his.compare_histogram_to_history(box_histogram, self.tracker_histogram_history[tracker_index])
            
            position_comparison = 0
            for position in self.tracker_position_history[tracker_index]:
                diff = math.sqrt((box_center[0] - position[0]) ** 2 + (box_center[1] - position[1]) ** 2)
                position_comparison += diff
            position_comparison /= len(self.tracker_position_history[tracker_index])
            position_comparison = position_comparison / (frames_passed + 1)
            
            size_comparison = 0
            for size in self.tracker_box_size_history[tracker_index]:
                diff = math.sqrt((w - size[0]) ** 2 + (h - size[1]) ** 2)
                size_comparison += diff
            size_comparison /= len(self.tracker_box_size_history[tracker_index])

            match_score[i] = hist_factor * histogram_comparison + pos_factor * position_comparison + size_factor * size_comparison
        
        if len(matching_boxes) == 0:
            return False, (0,0,0,0)
        
        best_match = np.argmin(match_score)
        x, y, x2, y2 = matching_boxes[best_match]
        return True, [x, y, x2-x, y2-y]

    def handle_overlapping(self, results, boxes):
        # if boxes are overlapping we can choose which one is the best match and discard the others
        return results, boxes





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
        frame_copy = self.next_frame_color.copy()
        
        success = [False for i in range(len(self.boxes))]
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        for i, points in enumerate(self.points):
            # apply the flow to the points
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

            # find the minimal box that contains all the points
            x_min = self.next_frame.shape[1] + 1
            y_min = self.next_frame.shape[0] + 1
            x_max = -1
            y_max = -1
            center = [0, 0]
            for j, point in enumerate(self.points[i]):
                x, y = point
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                center[0] += x
                center[1] += y
                cv2.circle(frame_copy, (point[0], point[1]), 4, (0, 255, 0), -1)
            center[0] /= len(self.points[i])
            center[1] /= len(self.points[i])
            cv2.circle(frame_copy, (int(center[0]), int(center[1])), 8, (255, 0, 0), -1)
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (255, 0, 0), 4)

            # find the motion detector box with the center closest to the edges of the box
            # but not any box is fine, it must contain the center of the points found
            bg = BGSUB.apply(self.next_frame_color)
            bg = det.preprocess(bg)
            new_boxes = det.extract_boxes(bg)
            box_index = -1
            min_distance = -1
            for j, new_box in enumerate(new_boxes):
                new_center = [(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2]
                distance = math.sqrt((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2)
                if distance < min_distance or min_distance == -1:
                    if new_box[0] <= center[0] <= new_box[2] and new_box[1] <= center[1] <= new_box[3]:
                        min_distance = distance
                        box_index = j

            if box_index == -1:
                # means no motion detector box was suitable
                success[i] = False
                
                # compute the box as the mean between the old box and the new surrounding box
                # x1 = self.boxes[i][0]
                # y1 = self.boxes[i][1]
                # x2 = self.boxes[i][2]
                # y2 = self.boxes[i][3]
                # old_box_factor = 0.6
                # points_box_factor = 0.4
                # mean_x1 = int(points_box_factor*x_min + old_box_factor*x1)
                # mean_y1 = int(points_box_factor*y_min + old_box_factor*y1)
                # mean_x2 = int(points_box_factor*x_max + old_box_factor*x2)
                # mean_y2 = int(points_box_factor*y_max + old_box_factor*y2)
                # cv2.rectangle(frame_copy, (mean_x1, mean_y1), (mean_x2, mean_y2), (0, 255, 255), 4)
                # self.boxes[i] = [mean_x1, mean_y1, mean_x2, mean_y2]

                # # resample the points that are not inside of the box
                # for j, point in enumerate(self.points[i]):
                #     x, y = point
                #     if not mean_x1 <= x <= mean_x2 or not mean_y1 <= y <= mean_y2:
                #         self.points[i][j] = self.sample_points(self.boxes[i], single_point_resample=True)
            
            if box_index != -1:
                # means that a motion detector box was found
                success[i] = True

                # the new box size will be the mean between the surrounding box size, the previous box size and the motion detector box size
                x1 = self.boxes[i][0]
                y1 = self.boxes[i][1]
                x2 = self.boxes[i][2]
                y2 = self.boxes[i][3]
                new_x1 = new_boxes[box_index][0]
                new_y1 = new_boxes[box_index][1]
                new_x2 = new_boxes[box_index][2]
                new_y2 = new_boxes[box_index][3]
                cv2.rectangle(frame_copy, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 4)
                motion_box_factor = 0.35
                points_box_factor = 0.25
                old_box_factor = 0.4
                mean_x1 = int(points_box_factor*x_min + old_box_factor*x1 + motion_box_factor*new_x1)
                mean_y1 = int(points_box_factor*y_min + old_box_factor*y1 + motion_box_factor*new_y1)
                mean_x2 = int(points_box_factor*x_max + old_box_factor*x2 + motion_box_factor*new_x2)
                mean_y2 = int(points_box_factor*y_max + old_box_factor*y2 + motion_box_factor*new_y2)
                cv2.rectangle(frame_copy, (mean_x1, mean_y1), (mean_x2, mean_y2), (0, 255, 0), 4)

                self.boxes[i] = [mean_x1, mean_y1, mean_x2, mean_y2]
            
                # resample the points that are not inside of the box
                for j, point in enumerate(self.points[i]):
                    x, y = point
                    if not mean_x1 <= x <= mean_x2 or not mean_y1 <= y <= mean_y2:
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