import cv2
import numpy as np
from ultralytics import YOLO

class RewardFunction:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.previous_area = 0
        self.previous_dis = 0
        self.prev_gray = None

    def calculate_area(self, bbox):
        # 計算偵測到的人像大小
        x1, y1, x2, y2 = bbox[:4]
        return (x2 - x1) * (y2 - y1)

    def calculate_distance(self, orig_shape, bbox):
        # 計算人像到畫面中央的距離
        x1, y1, x2, y2 = bbox[:4]
        x1, y1 = map(int, ((x1+x2)/2, (y1+y2)/2))  # 偵測到的人像中心點
        x2, y2 = map(int, (orig_shape[0]/2, orig_shape[1]/2))  # 畫面的中心點
        distance = ((x1-x2)**2 + (y1-y2)**2)**0.5  # 兩中心點的距離
        return distance

    def get_reward(self, results):
        frame = results.orig_img  # 初始化 frame 變數
        person_detected = False
        if results.boxes.shape[0] > 0:
            for cls in results.boxes.cls:
                if int(cls) == 0:  # 如果偵測到的物件類別為人(類別編號為0)
                    person_detected = True
                    break

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
                    print("moved")
                else:
                    print("not moved")
        self.prev_gray = gray

        current_area = 0
        # Handle detections using results.boxes
        if person_detected and self.previous_area != 0 and self.previous_dis != 0:
            for bbox, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls) == 0:  # 偵測到人標籤為0
                    area = self.calculate_area(bbox)
                    current_area += area
                    if area > self.previous_area:  # 偵測到的人像大小比之前大給予獎勵
                        reward += area * 0.01
                    elif area < self.previous_area:  # 偵測到的人像大小比之前小給予懲罰
                        reward -= area * 0.01
                    person_detected = True

            orig_shape = results[0].boxes.orig_shape  # 原圖像形狀
            current_dis = self.calculate_distance(orig_shape, bbox)  # 計算人像中心點到畫面中心點的距離
            if current_dis < self.previous_dis:  # 距離比之前更小給予獎勵
                reward += current_dis * 0.01
            elif current_dis > self.previous_dis:  # 距離比之前更大給予懲罰
                reward -= current_dis * 0.01
            self.previous_area = current_area
            self.previous_dis = current_dis

        if person_detected:
            reward += 1  # Base reward for detecting a person

        return reward