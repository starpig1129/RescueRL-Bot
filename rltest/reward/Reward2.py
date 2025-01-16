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
        self.target_rewards = {}      # 使用字典管理目標獎勵狀態
        self.TOUCH_THRESHOLD = 5.0    # 碰觸判定閾值

        # 移動距離歷史記錄
        self.movement_history = []    # 儲存最近150步的移動距離
        
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
        計算實體距離相關的獎勵，忽略高度差距，只考慮水平面（x-z平面）的距離
        """
        # 處理 point1 為字典的情況
        if isinstance(point1, dict):
            x1, z1 = float(point1['x']), float(point1['z'])
        else:  # 處理 numpy array 的情況
            x1, z1 = float(point1[0]), float(point1[1])
            
        # 處理 point2 為字典或 numpy array 的情況
        if isinstance(point2, dict):
            x2, z2 = float(point2['x']), float(point2['z'])
        else:  # numpy array
            x2, z2 = float(point2[0]), float(point2[1])

        # 計算水平面上的距離
        current_dis = ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5
        
        # 使用距離的相對變化計算獎勵
        if self.previous_dis > 0:
            distance_change = (self.previous_dis - current_dis) / self.previous_dis
            # 增加接近目標的獎勵
            dis_reward = max(0, distance_change)
            # 減少遠離目標的懲罰
            dis_punish = min(0, distance_change)  
        else:
            dis_reward = 0
            dis_punish = 0
        
        self.previous_dis = current_dis
        return dis_reward, dis_punish

    def move_reward(self, point1):
        """
        計算移動相關的獎勵，使用 Rigidbody position 計算實際移動距離
        """
        x1, z1 = point1['x'], point1['z']
        last_x, last_z = self.lastposition[0], self.lastposition[2]
        
        # 計算實際的物理移動距離
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
            move_punish = -1 / (1 + np.exp((total_movement - 5) / 2))
        else:
            move_reward = 0
            move_punish = 0

        self.lastposition = [x1, self.lastposition[1], z1]
        return move_reward, move_punish

    def is_up(self, rotation):
        """
        計算姿態相關的懲罰，使用 Rigidbody rotation 歐拉角度
        """
        # 獲取 x 和 z 軸的旋轉角度
        x_rotation = rotation.get('x', 0) % 360
        z_rotation = rotation.get('z', 0) % 360
        
        # 計算與正常姿態的偏差（正常姿態應該是x和z接近0度）
        x_deviation = min(x_rotation, 360 - x_rotation)
        z_deviation = min(z_rotation, 360 - z_rotation)
        
        # 綜合考慮兩個軸的偏差，當任一軸偏差超過45度時給予懲罰
        max_deviation = max(x_deviation, z_deviation)
        if max_deviation > 45:
            # 將偏差映射到0-1的懲罰值
            normalized_punishment = (max_deviation - 45) / 135  # (180-45=135)
            return min(1.0, normalized_punishment)
        return 0

    def touch(self, is_touch, crawler_pos):
        """
        計算碰觸目標的獎勵，只考慮水平面（x-z平面）的距離
        """
        touch_reward = 0
        con_reward = 0

        for i, target_pos in enumerate(is_touch):
            target_id = f"target_{i}"
            
            # 初始化目標獎勵狀態
            if target_id not in self.target_rewards:
                self.target_rewards[target_id] = True
                
            # 檢查是否可獲得獎勵
            if self.target_rewards[target_id]:
                # 計算相對距離（水平面）
                dx = target_pos['x'] - crawler_pos['x']
                dz = target_pos['z'] - crawler_pos['z']
                distance = np.sqrt(dx**2 + dz**2)
                print(distance)
                # 判定是否碰觸（使用水平距離）
                if distance < self.TOUCH_THRESHOLD:
                    self.target_rewards[target_id] = False
                    touch_reward = 1
                    self.con_touch_reward = 100  # 重置持續獎勵
                    break
        
        # 動態調整持續獎勵
        if self.con_touch_reward > 0:
            # 計算剩餘未碰觸目標數
            remaining_targets = sum(1 for v in self.target_rewards.values() if v)
            # 根據剩餘目標調整衰減率
            decay_rate = 0.95 + (remaining_targets * 0.01)
            
            con_reward = self.con_touch_reward
            self.con_touch_reward *= decay_rate
            
            if self.con_touch_reward < 1:
                self.con_touch_reward = 0

        return touch_reward, con_reward

    def target(self, reward_data):
        """
        找出最近的目標，只考慮水平面（x-z平面）的距離
        """
        # 只使用 x 和 z 座標
        crawler_position = np.array([
            reward_data['position']['x'],
            reward_data['position']['z']
        ])
        
        targets = reward_data['targets']
        target_positions = [
            np.array([t['position']['x'], t['position']['z']])
            for t in targets
        ]
        
        # 使用歐幾里德距離計算水平面距離
        distances = [np.linalg.norm(crawler_position - pos) for pos in target_positions]
        nearest_idx = np.argmin(distances)
        
        # 返回完整的目標位置和螢幕座標
        return target_positions[nearest_idx], targets[nearest_idx]['screenPosition']

    def get_reward(self, results, reward_data, angle):
        """
        計算總體獎勵值，調整獎勵權重
        """
        crawler_pos = reward_data['position']
        target_pos, target_view_pos = self.target(reward_data)
        rotation = reward_data['rotation']
        is_touch = [target['position'] for target in reward_data['targets']]
        
        person_detec_reward = self.person_detec_reward(results)
        dis_reward, dis_punish = self.distance_reward(crawler_pos, target_pos)
        inview_reward, viewdis_reward, viewdis_punish, everview_punish = self.person_in_view_reward(target_view_pos)
        move_reward, move_punish = self.move_reward(crawler_pos)
        upsidedown_punish = self.is_up(rotation)
        touch_reward, con_reward = self.touch(is_touch, crawler_pos)

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
            con_reward * 0                # 持續碰觸獎勵
        ]
        
        return np.sum(reward_list), reward_list
