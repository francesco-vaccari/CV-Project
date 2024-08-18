import os
import numpy as np
from itertools import combinations

class FrameAnnotation:
    def __init__(self, frame, team, player, x1, y1, x2, y2):
        self.frame = frame
        self.team = team
        self.player = player
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class Annotation:
    def __init__(self, annotations_folder, n_teams = 2, n_players_per_team = 6):
        annotations_files = [os.path.join(annotations_folder, file) for file in os.listdir(annotations_folder) if file.endswith('.label')]
        self.annotations = []
        for i in range(n_teams):
            self.annotations.append([])
            for j in range(n_players_per_team):
                self.annotations[i].append([])
        self.load_annotations(annotations_files)
    
    def load_annotations(self, annotations_files):
        for file in annotations_files:
            team, player = file.split('_')[1], file.split('_')[2].split('.')[0]
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    frame, x1, y1, x2, y2 = line.strip().split(',')
                    frame = int(frame)-1
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    self.annotations[int(team)-1][int(player)-1].append(FrameAnnotation(frame, team, player, x1, y1, x2, y2)) # frame number is the same as the index

    def get_annotation(self, n_frame):
        annotations = []
        for team in self.annotations:
            for player in team:
                annotations.append(player[n_frame])
        return annotations

    def evaluate(self, boxes, n_frame, threshold=0.5):
        if len(boxes) == 0:
            return 0, 0, []
        
        annotations = self.get_annotation(n_frame)
        tp = 0 # true positives
        fn = 0 # false negatives
        fp = 0 # false positives
        
        matches = {}
        index = 0
        for i, gt in enumerate(annotations):
            x1, y1, x2, y2 = gt.x1, gt.y1, gt.x2, gt.y2
            for j, pred in enumerate(boxes):
                pred_x1, pred_y1, pred_x2, pred_y2 = pred
                iou = self.calculate_iou([x1, y1, x2, y2], [pred_x1, pred_y1, pred_x2, pred_y2])
                if iou > threshold:
                    matches[index] = (i, j)
                    index += 1
        
        if index == 0:
            return 0, 0, []

        best_n_matches = 0
        annotations_used = []
        preds_used = []
        possible_matches = self.all_combinations(list(range(index)))
        for possible_match in possible_matches:
            n_matches = 0
            annotations_indexes = []
            preds_indexes = []
            for match in possible_match:
                i, j = matches[match]
                if i not in annotations_indexes and j not in preds_indexes:
                    n_matches += 1
                    annotations_indexes.append(i)
                    preds_indexes.append(j)
            
            if n_matches > best_n_matches:
                best_n_matches = n_matches
                annotations_used = annotations_indexes
                preds_used = preds_indexes
        
        tp = best_n_matches
        fn = len(annotations) - best_n_matches
        fp = len(boxes) - best_n_matches

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0

        return precision, recall, preds_used

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        except ZeroDivisionError:
            iou = 0
        
        return iou

    def all_combinations(self, nums):
        result = []
        for r in range(len(nums) + 1):
            result.extend([list(comb) for comb in combinations(nums, r)])
        return result
