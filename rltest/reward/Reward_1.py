import cv2
import numpy as np
from ultralytics import YOLO

class RewardFunction:
    def __init__(self):
        self.previous_area = 0
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    def calculate_area(self, bbox):
        # bbox is a tensor with shape (4,) -> [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox[:4]
        return (x2 - x1) * (y2 - y1)

    def get_reward(self, results):
        frame = results.orig_img
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reward = 0

        # Compute optical flow to determine if there was movement
        if self.prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                movement = np.linalg.norm(good_new - good_old, axis=1).mean()
                if movement > 1.0:  # Set a threshold for significant movement
                    reward += 0.1  # Movement reward
        self.prev_gray = gray

        person_detected = False
        current_area = 0

        # Handle detections using results.boxes
        if results.boxes is not None and len(results.boxes.cls) > 0:
            for bbox, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) == 0:  # Assuming class_id 0 is 'person'
                    area = self.calculate_area(bbox)
                    current_area += area
                    if area > self.previous_area:
                        reward += area * 0.01  # Increase reward as person size increases
                    elif area < self.previous_area:
                        reward -= area * 0.01  # Penalize if person size decreases
                    person_detected = True

        if person_detected:
            reward += 1  # Base reward for detecting a person

        self.previous_area = current_area

        return reward



