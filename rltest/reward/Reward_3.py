import cv2
import numpy as np
from ultralytics import YOLO

class RewardFunction:
    def __init__(self):
        self.pre_viewdis = 0
        self.previous_dis = 0
        self.everview = True
        self.lastposition = [0,0,0]
        self.con_touch_reward = 0
        self.target1_reward = True
        self.target2_reward = True
        self.target3_reward = True
    
    def person_detec_reward(self, results):# 人像偵測獎勵    
        person_detec_reward = 0
        if len(results) == 1:
            if results[0].boxes.cls == 0:
                person_detec_reward = 1
        return person_detec_reward
    
    def person_in_view_reward(self, view): # 人物位置偵測
        inview_reward = 0
        viewdis_reward = 0
        viewdis_punish = 0
        everview_punish = 0

        # 假設 view 是字典，包含 'x' 和 'y' 鍵
        if view.get('x', 0.0) != 0.0 and view.get('y', 0.0) != 0.0: # 在視野範圍內
            self.everview = True
            inview_reward = 1
            current_viewdis = ((view['x'] - 0.5) ** 2 + (view['y'] - 0.5) ** 2) ** 0.5 # 人物到視野中心距離
            if current_viewdis <= 0.2: # 在畫面中心處
                if current_viewdis < self.pre_viewdis: # 靠近畫面中心
                    viewdis_reward = 1
            else: # 不在畫面中心
                if current_viewdis < self.pre_viewdis: # 靠近畫面中心
                    viewdis_reward = 1
                elif current_viewdis > self.pre_viewdis: # 遠離畫面中心
                    viewdis_punish = -1
            self.pre_viewdis = current_viewdis
        else: # 不在視野範圍內
            inview_reward = 0
            if self.everview: # 曾經在視野範圍內(丟失人物視野)
                everview_punish = -1 

        return inview_reward, viewdis_reward, viewdis_punish, everview_punish

    
    def distance_reward(self, point1, point2):
        dis_reward = 0
        dis_punish = 0
        
        # Extract coordinates from point1 (dictionary)
        x1, y1, z1 = float(point1['x']), float(point1['y']), float(point1['z'])
        
        # Extract coordinates from point2 (list or array)
        x2, y2, z2 = float(point2[0]), float(point2[1]), float(point2[2])

        # Calculate the distance
        current_dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        
        if current_dis < self.previous_dis:  # 接近人
            dis_reward = 1
        elif current_dis > self.previous_dis:  # 遠離人
            dis_punish = -1
        
        self.previous_dis = current_dis
        return dis_reward, dis_punish

    
    def move_reward(self, point1):  # 移動獎勵/懲罰
        move_punish = 0
        move_reward = 0

        # 提取 x 和 z 坐標值（假設 x, z 是移動時的重要坐標）
        x1, z1 = point1['x'], point1['z']
        last_x, last_z = self.lastposition[0], self.lastposition[2]

        # 計算移動距離
        movedis = ((x1 - last_x) ** 2 + (z1 - last_z) ** 2) ** 0.5  # 距離應該是平方根而不是次方
        
        # 判斷移動距離是否太小（視為停滯）
        if movedis < 0.3:
            move_punish = -1
        else:
            move_reward = 1

        # 更新最後位置
        self.lastposition = [x1, self.lastposition[1], z1]
        
        return move_reward, move_punish

        
    def is_up(self, up):  # 上下顛倒懲罰
        upsidedown_punish = 0

        # 假設 up 是字典，並且 'y' 是我們要檢查的鍵
        if up.get('y', 1) < 0:  # 預設值為 1，防止找不到 'y' 時的 KeyError
            upsidedown_punish = 1
        
        return upsidedown_punish

    
    def touch(self, is_touch):  # 碰到人獎勵
        touch_reward = 0
        con_reward = 0

        # 假設 is_touch 是一個包含字典的列表，且每個字典中有 'x', 'y', 'z' 坐標值
        if is_touch[0]['x'] < -5 and self.target1_reward:
            self.target1_reward = False
            touch_reward = 1
        elif is_touch[1]['x'] < -5 and self.target2_reward:
            self.target2_reward = False
            touch_reward = 1
        elif is_touch[2]['x'] < -5 and self.target3_reward:
            self.target3_reward = False
            touch_reward = 1

        # 持續獎勵遞減
        if touch_reward:
            self.con_touch_reward = 100
        
        if self.con_touch_reward >= 5:
            self.con_touch_reward -= 5
            con_reward = self.con_touch_reward

        return touch_reward, con_reward

    
    def target(self, reward_data):
        # Assuming reward_data contains the position of the crawler and the targets in a dictionary
        crawler_position = np.array([reward_data['position']['x'], reward_data['position']['y'], reward_data['position']['z']])  # Extract the crawler's position as an array
        targets = reward_data['targets']  # Get the list of targets
        
        # Ensure each target's 'position' is a list or array, not a dict
        target1_position = np.array([targets[0]['position']['x'], targets[0]['position']['y'], targets[0]['position']['z']])
        target2_position = np.array([targets[1]['position']['x'], targets[1]['position']['y'], targets[1]['position']['z']])
        target3_position = np.array([targets[2]['position']['x'], targets[2]['position']['y'], targets[2]['position']['z']])

        # Calculate distances to each target
        target1_distance = np.linalg.norm(crawler_position - target1_position)
        target2_distance = np.linalg.norm(crawler_position - target2_position)
        target3_distance = np.linalg.norm(crawler_position - target3_position)

        # Find the closest target
        if target1_distance < target2_distance and target1_distance < target3_distance:
            point2 = target1_position
            view = targets[0]['screenPosition']
        elif target2_distance < target1_distance and target2_distance < target3_distance:
            point2 = target2_position
            view = targets[1]['screenPosition']
        else:
            point2 = target3_position
            view = targets[2]['screenPosition']
        
        return point2, view

        
    def get_reward(self, results, reward_data):
        # Assume reward_data is a dictionary with keys 'position', 'rotation', and 'targets'
        point1 = reward_data['position']  # Crawler position
        point2, view = self.target(reward_data)  # Human position / human screen position  
        up = reward_data['rotation']  # Crawler standing/rotation information
        is_touch = [target['position'] for target in reward_data['targets']]  # Touch information for each target

        # Calculate individual reward components
        person_detec_reward = self.person_detec_reward(results)  # Person detection reward
        dis_reward, dis_punish = self.distance_reward(point1, point2)  # Crawler position reward/punishment
        inview_reward, viewdis_reward, viewdis_punish, everview_punish = self.person_in_view_reward(view)  # Person view reward/punishment
        move_reward, move_punish = self.move_reward(point1)  # Movement reward/punishment
        upsidedown_punish = self.is_up(up)  # Upside-down punishment        
        touch_reward, con_reward = self.touch(is_touch)  # Touch reward

        # Aggregate all rewards into a final reward
        reward = (person_detec_reward * 1 +      # Person detection reward
                dis_reward          * 10 +      # Crawler position approaching person reward
                dis_punish          * 5 +      # Crawler position moving away from person punishment
                inview_reward       * 2 +      # Person in view reward
                viewdis_reward      * 4 +      # Person moving closer to center of view reward
                viewdis_punish      * 5 +      # Person moving away from center of view punishment
                everview_punish     * 10 +     # Lost person from view punishment
                move_reward         * 0.5 +    # Movement reward
                move_punish         * 0.25 +   # Low movement punishment (stuck)
                upsidedown_punish   * 10 +     # Upside-down punishment
                touch_reward        * 100 +    # Touch reward
                con_reward          * 1 )      # Continuous touch reward

        return reward,[person_detec_reward,dis_reward,dis_punish,inview_reward,viewdis_reward,viewdis_punish,everview_punish,move_reward,move_punish,upsidedown_punish,touch_reward,con_reward]
