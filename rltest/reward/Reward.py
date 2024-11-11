import numpy as np

class RewardFunction:
    """
    獎勵函數類別，用於計算強化學習環境中的獎勵值
    包含多個子獎勵函數，最終將所有獎勵組合成總獎勵
    """
    def __init__(self):
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
    
    def person_detec_reward(self, results):
        """
        計算人像偵測獎勵
        Args:
            results: YOLO模型的偵測結果
        Returns:
            person_detec_reward: 若偵測到人像則為1，否則為0
        """
        person_detec_reward = 0
        if len(results) == 1 and results[0].boxes.cls == 0:
            person_detec_reward = 1
        return person_detec_reward
    
    def person_in_view_reward(self, view):
        """
        計算目標是否在視野內的相關獎勵
        Args:
            view: 包含目標在螢幕上x,y座標的字典
        Returns:
            tuple: (在視野內獎勵, 距離變近獎勵, 距離變遠懲罰, 失去視野懲罰)
        """
        inview_reward = 0     # 在視野內的獎勵
        viewdis_reward = 0    # 距離變近的獎勵
        viewdis_punish = 0    # 距離變遠的懲罰
        everview_punish = 0   # 失去視野的懲罰

        # 檢查目標是否在視野範圍內
        if view.get('x', 0.0) != 0.0 and view.get('y', 0.0) != 0.0:
            self.everview = True
            inview_reward = 1
            
            # 計算目標到視野中心的距離
            current_viewdis = ((view['x'] - 0.5) ** 2 + (view['y'] - 0.5) ** 2) ** 0.5
            
            # 根據距離變化給予獎勵或懲罰
            if current_viewdis <= 0.2:  # 目標在畫面中心附近
                if current_viewdis < self.pre_viewdis:
                    viewdis_reward = 1
            else:  # 目標不在畫面中心
                if current_viewdis < self.pre_viewdis:
                    viewdis_reward = 1
                elif current_viewdis > self.pre_viewdis:
                    viewdis_punish = -1
                    
            self.pre_viewdis = current_viewdis
        else:  # 目標不在視野範圍內
            inview_reward = 0
            if self.everview:  # 如果之前看過目標，給予失去視野的懲罰
                everview_punish = -1 

        return inview_reward, viewdis_reward, viewdis_punish, everview_punish

    def distance_reward(self, point1, point2):
        """
        計算實體距離相關的獎勵
        Args:
            point1: 當前位置的座標字典
            point2: 目標位置的座標列表
        Returns:
            tuple: (接近獎勵, 遠離懲罰)
        """
        dis_reward = 0    # 接近目標的獎勵
        dis_punish = 0    # 遠離目標的懲罰
        
        # 提取座標點
        x1, y1, z1 = float(point1['x']), float(point1['y']), float(point1['z'])
        x2, y2, z2 = float(point2[0]), float(point2[1]), float(point2[2])

        # 計算當前距離
        current_dis = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        
        # 根據距離變化給予獎勵或懲罰
        if current_dis < self.previous_dis:
            dis_reward = 1
        elif current_dis > self.previous_dis:
            dis_punish = -1
        
        self.previous_dis = current_dis
        return dis_reward, dis_punish

    def move_reward(self, point1):
        """
        計算移動相關的獎勵
        Args:
            point1: 當前位置的座標字典
        Returns:
            tuple: (移動獎勵, 停滯懲罰)
        """
        move_punish = 0   # 停滯的懲罰
        move_reward = 0   # 移動的獎勵

        # 計算水平面上的移動距離
        x1, z1 = point1['x'], point1['z']
        last_x, last_z = self.lastposition[0], self.lastposition[2]
        movedis = ((x1 - last_x) ** 2 + (z1 - last_z) ** 2) ** 0.5
        
        # 判斷是否停滯
        if movedis < 0.3:
            move_punish = -1
        else:
            move_reward = 1

        # 更新上一次位置
        self.lastposition = [x1, self.lastposition[1], z1]
        return move_reward, move_punish

    def is_up(self, up):
        """
        計算姿態相關的懲罰
        Args:
            up: 包含旋轉資訊的字典
        Returns:
            int: 翻倒懲罰值
        """
        upsidedown_punish = 0
        if up.get('y', 1) < 0:  # y值小於0表示翻倒
            upsidedown_punish = 1
        return upsidedown_punish

    def touch(self, is_touch):
        """
        計算碰觸目標的獎勵
        Args:
            is_touch: 包含各目標位置資訊的列表
        Returns:
            tuple: (碰觸獎勵, 持續獎勵)
        """
        touch_reward = 0  # 碰觸獎勵
        con_reward = 0    # 持續獎勵

        # 檢查是否碰觸到各個目標
        if is_touch[0]['x'] < -5 and self.target1_reward:
            self.target1_reward = False
            touch_reward = 1
        elif is_touch[1]['x'] < -5 and self.target2_reward:
            self.target2_reward = False
            touch_reward = 1
        elif is_touch[2]['x'] < -5 and self.target3_reward:
            self.target3_reward = False
            touch_reward = 1

        # 設定持續獎勵
        if touch_reward:
            self.con_touch_reward = 100
        
        # 持續獎勵遞減機制
        if self.con_touch_reward >= 5:
            self.con_touch_reward -= 5
            con_reward = self.con_touch_reward

        return touch_reward, con_reward

    def target(self, reward_data):
        """
        找出最近的目標
        Args:
            reward_data: 包含crawler位置和目標位置的字典
        Returns:
            tuple: (最近目標的位置, 最近目標的螢幕座標)
        """
        # 取得crawler位置
        crawler_position = np.array([
            reward_data['position']['x'],
            reward_data['position']['y'],
            reward_data['position']['z']
        ])
        
        targets = reward_data['targets']
        
        # 轉換目標座標為numpy數組
        target_positions = [
            np.array([t['position']['x'], t['position']['y'], t['position']['z']])
            for t in targets
        ]
        
        # 計算到各目標的距離
        distances = [np.linalg.norm(crawler_position - pos) for pos in target_positions]
        
        # 找出最近的目標
        nearest_idx = np.argmin(distances)
        
        return target_positions[nearest_idx], targets[nearest_idx]['screenPosition']

    def get_reward(self, results, reward_data):
        """
        計算總體獎勵值
        Args:
            results: YOLO模型的偵測結果
            reward_data: 包含位置、旋轉等資訊的字典
        Returns:
            tuple: (總獎勵值, 各分項獎勵值列表)
        """
        # 取得必要資訊
        point1 = reward_data['position']
        point2, view = self.target(reward_data)
        up = reward_data['rotation']
        is_touch = [target['position'] for target in reward_data['targets']]

        # 計算各項獎勵
        person_detec_reward = self.person_detec_reward(results)
        dis_reward, dis_punish = self.distance_reward(point1, point2)
        inview_reward, viewdis_reward, viewdis_punish, everview_punish = self.person_in_view_reward(view)
        move_reward, move_punish = self.move_reward(point1)
        upsidedown_punish = self.is_up(up)
        touch_reward, con_reward = self.touch(is_touch)

        # 計算總獎勵
        reward = (
            person_detec_reward * 1 +    # 人像偵測獎勵
            dis_reward * 10 +            # 接近目標獎勵
            dis_punish * 5 +             # 遠離目標懲罰
            inview_reward * 2 +          # 目標在視野內獎勵
            viewdis_reward * 4 +         # 目標接近視野中心獎勵
            viewdis_punish * 5 +         # 目標遠離視野中心懲罰
            everview_punish * 10 +       # 失去目標視野懲罰
            move_reward * 0.5 +          # 移動獎勵
            move_punish * 0.25 +         # 停滯懲罰
            upsidedown_punish * 10 +     # 翻倒懲罰
            touch_reward * 100 +         # 碰觸目標獎勵
            con_reward * 1               # 持續碰觸獎勵
        )

        # 回傳總獎勵和分項獎勵列表
        reward_list = [
            person_detec_reward, dis_reward, dis_punish,
            inview_reward, viewdis_reward, viewdis_punish,
            everview_punish, move_reward, move_punish,
            upsidedown_punish, touch_reward, con_reward
        ]
        
        return reward, reward_list