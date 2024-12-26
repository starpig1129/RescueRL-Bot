import numpy as np

class RewardFunction:
    """
    獎勵函數類別，用於計算環境中的獎勵值
    包含多個子獎勵函數，最終將所有獎勵組合成總獎勵
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """
        重置所有狀態變數
        """
        # 視角相關變數
        self.pre_viewdis = 0          # 前一幀的視野距離
        self.previous_dis = 0         # 前一幀的實體距離
        self.everview = True          # 是否曾經看見目標
        self.lastposition = [0,0,0]   # 上一次的位置
        
        # 碰觸獎勵相關變數
        self.con_touch_reward = 0     # 持續碰觸獎勵
        self.target1_reward = True    # 目標1是否可獲得獎勵
        self.target2_reward = True    # 目標2是否可獲得獎勵
        self.target3_reward = True    # 目標3是否可獲得獎勵

        # 移動距離歷史記錄
        self.movement_history = []    # 儲存最近200步的移動距離
        
        # 新增: 追蹤連續看見目標的次數
        self.continuous_view_count = 0

    def person_detec_reward(self, results):
        """
        計算人像偵測獎勵，使用連續性獎勵
        """
        if len(results) == 1 and results[0].boxes.cls == 0:
            confidence = float(results[0].boxes.conf[0])
            # 根據偵測信心度給予連續性獎勵
            return min(1.0, confidence)
        return 0

    def person_in_view_reward(self, view):
        """
        計算目標是否在視野內的相關獎勵，加入連續性和漸進式獎勵
        """
        inview_reward = 0     
        viewdis_reward = 0    
        viewdis_punish = 0    
        everview_punish = 0   

        if view.get('x', 0.0) != 0.0 and view.get('y', 0.0) != 0.0:
            self.everview = True
            self.continuous_view_count += 1
            
            # 連續看見目標的額外獎勵
            inview_reward = min(2.0, 1 + self.continuous_view_count / 100)
            
            # 計算目標到視野中心的距離
            current_viewdis = ((view['x'] - 0.5) ** 2 + (view['y'] - 0.5) ** 2) ** 0.5
            
            # 使用平滑的獎勵函數
            if current_viewdis <= 0.3:  # 放寬中心區域
                # 越接近中心獎勵越高
                center_reward = 1 - (current_viewdis / 0.3)
                if current_viewdis < self.pre_viewdis:
                    viewdis_reward = center_reward
            else:
                if current_viewdis < self.pre_viewdis:
                    viewdis_reward = 0.5  # 減少非中心區域的獎勵
                elif current_viewdis > self.pre_viewdis:
                    viewdis_punish = -0.5  # 減少懲罰強度
                    
            self.pre_viewdis = current_viewdis
        else:
            self.continuous_view_count = 0
            inview_reward = 0
            if self.everview:
                # 根據連續看見的時間調整懲罰
                everview_punish = -0.5  # 減輕失去視野的懲罰

        return inview_reward, viewdis_reward, viewdis_punish, everview_punish

    def distance_reward(self, point1, point2):
        """
        計算實體距離相關的獎勵，使用連續性獎勵函數
        """
        x1, y1, z1 = float(point1['x']), float(point1['y']), float(point1['z'])
        x2, y2, z2 = float(point2[0]), float(point2[1]), float(point2[2])

        current_dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        
        # 使用距離的相對變化計算獎勵
        if self.previous_dis > 0:
            distance_change = (self.previous_dis - current_dis) / self.previous_dis
            dis_reward = max(0, distance_change * 2)  # 放大獎勵
            dis_punish = min(0, distance_change)  # 保持原有懲罰
        else:
            dis_reward = 0
            dis_punish = 0
        
        self.previous_dis = current_dis
        return dis_reward, dis_punish

    def move_reward(self, point1):
        """
        計算移動相關的獎勵，使用自適應閾值
        """
        x1, z1 = point1['x'], point1['z']
        last_x, last_z = self.lastposition[0], self.lastposition[2]
        movedis = ((x1 - last_x) ** 2 + (z1 - last_z) ** 2) ** 0.5
        
        self.movement_history.append(movedis)
        if len(self.movement_history) > 150:
            self.movement_history.pop(0)
        
        # 計算移動距離的均值和標準差
        if len(self.movement_history) > 0:
            mean_movement = np.mean(self.movement_history)
            std_movement = np.std(self.movement_history)
            
            # 自適應移動閾值
            movement_threshold = mean_movement + std_movement
            
            # 連續性移動獎勵
            move_reward = min(1.0, movedis / movement_threshold)
            
            # 使用改進的sigmoid函數計算懲罰
            total_movement = sum(self.movement_history)
            move_punish = -1 / (1 + np.exp((total_movement - 5) / 2))  # 調整曲線斜率
        else:
            move_reward = 0
            move_punish = 0

        self.lastposition = [x1, self.lastposition[1], z1]
        return move_reward, move_punish

    def is_up(self, up):
        """
        計算姿態相關的懲罰，使用連續性懲罰
        """
        y_value = up.get('y', 1)
        if y_value < 0:
            # 根據傾斜程度給予不同程度的懲罰
            return min(1.0, abs(y_value))
        return 0

    def touch(self, is_touch):
        """
        計算碰觸目標的獎勵，使用指數衰減
        """
        touch_reward = 0
        con_reward = 0

        if is_touch[0]['x'] < -5 and self.target1_reward:
            self.target1_reward = False
            touch_reward = 1
            self.con_touch_reward = 100
        
        # 使用指數衰減而不是線性衰減
        if self.con_touch_reward > 0:
            con_reward = self.con_touch_reward
            self.con_touch_reward *= 0.95  # 每次衰減5%
            if self.con_touch_reward < 1:
                self.con_touch_reward = 0

        return touch_reward, con_reward

    def target(self, reward_data):
        """
        找出最近的目標
        """
        crawler_position = np.array([
            reward_data['position']['x'],
            reward_data['position']['y'],
            reward_data['position']['z']
        ])
        
        targets = reward_data['targets']
        target_positions = [
            np.array([t['position']['x'], t['position']['y'], t['position']['z']])
            for t in targets
        ]
        
        distances = [np.linalg.norm(crawler_position - pos) for pos in target_positions]
        nearest_idx = np.argmin(distances)
        
        return target_positions[nearest_idx], targets[nearest_idx]['screenPosition']

    def get_reward(self, results, reward_data, angle):
        """
        計算總體獎勵值，調整獎勵權重
        """
        point1 = reward_data['position']
        point2, view = self.target(reward_data)
        up = reward_data['rotation']
        is_touch = [target['position'] for target in reward_data['targets']]

        person_detec_reward = self.person_detec_reward(results)
        dis_reward, dis_punish = self.distance_reward(point1, point2)
        inview_reward, viewdis_reward, viewdis_punish, everview_punish = self.person_in_view_reward(view)
        move_reward, move_punish = self.move_reward(point1)
        upsidedown_punish = self.is_up(up)
        touch_reward, con_reward = self.touch(is_touch)

        # 調整獎勵權重
        reward_list = [
            person_detec_reward * 10,     # 人像偵測獎勵
            dis_reward * 15,              # 接近目標獎勵
            dis_punish * 7.5,             # 遠離目標懲罰
            inview_reward * 3,            # 目標在視野內獎勵
            viewdis_reward * 8,           # 目標接近視野中心獎勵
            viewdis_punish * 4,           # 目標遠離視野中心懲罰
            everview_punish * 8,          # 失去目標視野懲罰
            move_reward * 2,              # 移動獎勵
            move_punish * 8,              # 移動距離懲罰
            upsidedown_punish * 12,       # 翻倒懲罰
            touch_reward * 150,           # 碰觸目標獎勵
            con_reward * 1.5              # 持續碰觸獎勵
        ]
        
        return np.sum(reward_list), reward_list
