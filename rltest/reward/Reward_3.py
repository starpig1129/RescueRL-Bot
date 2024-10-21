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
    
    def person_in_view_reward(self, view): #人物位置偵測
        inview_reward = 0
        viewdis_reward = 0
        viewdis_punish = 0
        everview_punish = 0
        if view[0] != 0.0 and view[1] != 0.0 : #在視野範圍內
            self.everview = True
            inview_reward = 1
            current_viewdis = ((view[0]-0.5) ** 2 + (view[1] - 0.5) ** 2) ** 0.5 # 人物到視野中心距離
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
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        current_dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        if current_dis < self.previous_dis: # 接近人
            dis_reward = 1
        elif current_dis > self.previous_dis:# 遠離人
            dis_punish = -1
        self.previous_dis = current_dis
        return dis_reward, dis_punish
    
    def move_reward(self, point1): # 移動獎勵/懲罰
        move_punish = 0
        move_reward = 0
        movedis = ((point1[0] - self.lastposition[0]) ** 2 + (point1[2] - self.lastposition[2]) ** 2) ** 5
        if movedis < 0.3:
            move_punish = -1
        else:
            move_reward = 1
        self.lastposition = point1
        return move_reward, move_punish
        
    def is_up(self, up): # 上下顛倒懲罰
        upsidedown_punish = 0
        if up[1] < 0:
            upsidedown_punish = 1
        return upsidedown_punish
    
    def touch(self, is_touch): # 碰到人獎勵
        if is_touch[0] < -5 and self.target1_reward:
            is_touch = True
            self.target1_reward = False
        elif is_touch[1] < -5 and self.target2_reward:
            is_touch = True
            self.target2_reward = False
        elif is_touch[2] < -5 and self.target3_reward:
            is_touch = True
            self.target3_reward = False
        else:
            is_touch = False

        touch_reward = 0
        con_reward = 0
        if is_touch:
            touch_reward = 1
            self.con_touch_reward = 100  
        if self.con_touch_reward >= 5: # 碰到人後持續給遞減的獎勵
            self.con_touch_reward -= 5
            con_reward = self.con_touch_reward
        return touch_reward, con_reward
    
    def target(self, reward_data):
        target1=((reward_data[0]-reward_data[6])**2+(reward_data[2]-reward_data[8])**2)**0.5
        target2=((reward_data[0]-reward_data[11])**2+(reward_data[2]-reward_data[13])**2)**0.5
        target3=((reward_data[0]-reward_data[16])**2+(reward_data[2]-reward_data[18])**2)**0.5
        if target1 < target2 and target1 < target3:
            point2 = reward_data[6:9]
            view = reward_data[9:11]
        elif target2 < target1 and target2 < target3:
            point2 = reward_data[11:14]
            view = reward_data[14:16]
        elif target3 < target2 and target3 < target1:
            point2 = reward_data[16:19]
            view = reward_data[19:21]
        return point2, view
        
    def get_reward(self, results, reward_data):
        point1 = reward_data[:3] # crawler位置
        point2, view = self.target(reward_data) # 人物位置/人物畫面位置  
        up = reward_data[3:6]  # crawler站立/翻倒資訊
        is_touch = [reward_data[7],reward_data[12],reward_data[17]]
        
        person_detec_reward = self.person_detec_reward(results) # 人像偵測獎勵
        dis_reward, dis_punish = self.distance_reward(point1, point2) # Crawler位置獎勵/懲罰
        inview_reward, viewdis_reward, viewdis_punish, everview_punish = self.person_in_view_reward(view) # 人像偵測獎勵/懲罰
        move_reward, move_punish = self.move_reward(point1) # 移動獎勵/懲罰 
        upsidedown_punish = self.is_up(up) # 上下顛倒懲罰        
        touch_reward, con_reward = self.touch(is_touch) # 觸碰獎勵
        
        reward = 0
        reward = (person_detec_reward * 1 +      # 人像偵測獎勵
                  dis_reward          * 10 +      # Crawler位置接近人獎勵
                  dis_punish          * 5 +      # Crawler位置遠離人懲罰
                  inview_reward       * 2 +      # 人物在視野範圍內獎勵
                  viewdis_reward      * 4 +      # 人物靠近視野中心獎勵
                  viewdis_punish      * 5 +      # 人物遠離視野中心懲罰
                  everview_punish     * 10 +      # 丟失人物視野懲罰
                  move_reward         * 0.5 +      # 移動獎勵
                  move_punish         * 0.25 +      # 移動過低(卡住)懲罰
                  upsidedown_punish   * 10 +      # 上下顛倒懲罰
                  touch_reward        * 100 +      # 觸碰獎勵
                  con_reward          * 1 )     # 觸碰後的持續獎勵(先不用管) 
        return reward
