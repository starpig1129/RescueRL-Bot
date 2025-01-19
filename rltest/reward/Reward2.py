"""獎勵函數模組，用於計算強化學習環境中的獎勵值。

此模組實現了一個獎勵函數類別，用於計算智能體在環境中的各種獎勵值。
包含多個子獎勵函數，最終將所有獎勵組合成總獎勵。
"""

from typing import Dict, List, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray

class RewardFunction:
    """獎勵函數類別，計算環境中的獎勵值。
    
    此類別包含多個子獎勵函數，用於評估智能體的不同行為表現，
    並將這些子獎勵組合成最終的總獎勵值。
    
    屬性:
        TOUCH_THRESHOLD: float, 碰觸判定的距離閾值
        MAX_MOVEMENT_HISTORY: int, 移動歷史記錄的最大長度
        ROTATION_THRESHOLD: float, 姿態偏差的閾值(度)
        VIEW_CENTER_THRESHOLD: float, 視野中心區域的閾值
    """
    
    # 類別常數
    TOUCH_THRESHOLD: float = 15.0
    MAX_MOVEMENT_HISTORY: int = 150
    ROTATION_THRESHOLD: float = 45.0
    VIEW_CENTER_THRESHOLD: float = 0.3
    
    # 獎勵權重常數
    DETECTION_WEIGHT: float = 10.0
    APPROACH_WEIGHT: float = 15.0
    LEAVE_PENALTY_WEIGHT: float = 7.5
    IN_VIEW_WEIGHT: float = 3.0
    CENTER_APPROACH_WEIGHT: float = 8.0
    CENTER_LEAVE_WEIGHT: float = 4.0
    LOST_VIEW_WEIGHT: float = 8.0
    MOVEMENT_WEIGHT: float = 2.0
    MOVEMENT_PENALTY_WEIGHT: float = 8.0
    POSTURE_PENALTY_WEIGHT: float = 12.0
    TOUCH_WEIGHT: float = 150.0
    CONTINUOUS_TOUCH_WEIGHT: float = 0.0
    
    def __init__(self):
        """初始化獎勵函數類別。"""
        self.reset()
        
    def reset(self) -> None:
        """重置所有狀態變數。"""
        # 視角相關變數
        self._prev_view_distance: float = 0.0  # 前一幀的視野距離
        self._prev_entity_distance: float = 0.0  # 前一幀的實體距離
        self._has_seen_target: bool = True  # 是否曾經看見目標
        self._last_position: List[float] = [0.0, 0.0, 0.0]  # 上一次的位置
        
        # 碰觸獎勵相關變數
        self._continuous_touch_reward: float = 0.0  # 持續碰觸獎勵
        self._target_reward_states: Dict[str, bool] = {}  # 目標獎勵狀態管理
        
        # 移動距離歷史記錄
        self._movement_history: List[float] = []  # 儲存最近移動距離
        
        # 連續看見目標的計數
        self._continuous_view_count: int = 0

    def calculate_detection_reward(self, detection_results: List[Any]) -> float:
        """計算人像偵測獎勵。
        
        根據偵測結果的信心度給予連續性獎勵。
        
        參數:
            detection_results: YOLO偵測結果列表
            
        回傳:
            float: 偵測獎勵值 (0.0-1.0)
        """
        if len(detection_results) == 1 and detection_results[0].boxes.cls == 0:
            confidence = float(detection_results[0].boxes.conf[0])
            return min(1.0, confidence)
        return 0.0

    def calculate_view_rewards(self, view: Dict[str, float]) -> Tuple[float, float, float, float]:
        """計算目標在視野內的相關獎勵。
        
        根據目標在視野中的位置計算多個獎勵值，包括:
        - 目標在視野內的基本獎勵
        - 目標接近視野中心的獎勵
        - 目標遠離視野中心的懲罰
        - 失去目標視野的懲罰
        
        參數:
            view: 包含目標在螢幕上x,y座標的字典
            
        回傳:
            Tuple[float, float, float, float]: 
            (視野內獎勵, 接近中心獎勵, 遠離中心懲罰, 失去視野懲罰)
        """
        in_view_reward = 0.0
        center_approach_reward = 0.0
        center_leave_penalty = 0.0
        lost_view_penalty = 0.0

        if view.get('x', 0.0) != 0.0 and view.get('y', 0.0) != 0.0:
            self._has_seen_target = True
            self._continuous_view_count += 1
            
            # 連續看見目標的漸進式獎勵
            in_view_reward = min(2.0, 1.0 + self._continuous_view_count / 100.0)
            
            # 計算目標到視野中心的距離
            current_view_distance = np.sqrt(
                (view['x'] - 0.5) ** 2 + (view['y'] - 0.5) ** 2
            )
            
            # 根據目標位置給予獎勵或懲罰
            if current_view_distance <= self.VIEW_CENTER_THRESHOLD:
                center_reward = 1.0 - (current_view_distance / self.VIEW_CENTER_THRESHOLD)
                if current_view_distance < self._prev_view_distance:
                    center_approach_reward = center_reward
            else:
                if current_view_distance < self._prev_view_distance:
                    center_approach_reward = 0.5
                elif current_view_distance > self._prev_view_distance:
                    center_leave_penalty = -0.5
                    
            self._prev_view_distance = current_view_distance
        else:
            self._continuous_view_count = 0
            if self._has_seen_target:
                lost_view_penalty = -0.5

        return in_view_reward, center_approach_reward, center_leave_penalty, lost_view_penalty

    def _extract_coordinates(self, point: Union[Dict[str, float], NDArray]) -> Tuple[float, float]:
        """從不同格式的點資料中提取x,z座標。
        
        參數:
            point: 包含座標的字典或numpy陣列
            
        回傳:
            Tuple[float, float]: (x座標, z座標)
        """
        if isinstance(point, dict):
            return float(point['x']), float(point['z'])
        return float(point[0]), float(point[1])
        
    def calculate_distance_rewards(self, point1: Union[Dict[str, float], NDArray], 
                                 point2: Union[Dict[str, float], NDArray]) -> Tuple[float, float]:
        """計算實體距離相關的獎勵。
        
        忽略高度差距，只考慮水平面(x-z平面)的距離變化。
        
        參數:
            point1: 第一個點的座標
            point2: 第二個點的座標
            
        回傳:
            Tuple[float, float]: (接近獎勵, 遠離懲罰)
        """
        x1, z1 = self._extract_coordinates(point1)
        x2, z2 = self._extract_coordinates(point2)

        current_distance = np.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
        
        if self._prev_entity_distance > 0:
            distance_change = (self._prev_entity_distance - current_distance) / self._prev_entity_distance
            approach_reward = max(0.0, distance_change)
            leave_penalty = min(0.0, distance_change)
        else:
            approach_reward = 0.0
            leave_penalty = 0.0
        
        self._prev_entity_distance = current_distance
        return approach_reward, leave_penalty

    def calculate_movement_rewards(self, current_position: Dict[str, float]) -> Tuple[float, float]:
        """計算移動相關的獎勵。
        
        使用Rigidbody position計算實際移動距離，並根據歷史移動記錄
        給予適當的獎勵和懲罰。
        
        參數:
            current_position: 當前位置座標
            
        回傳:
            Tuple[float, float]: (移動獎勵, 移動懲罰)
        """
        x1, z1 = current_position['x'], current_position['z']
        last_x, last_z = self._last_position[0], self._last_position[2]
        
        # 計算移動距離
        move_distance = np.sqrt((x1 - last_x) ** 2 + (z1 - last_z) ** 2)
        
        # 更新移動歷史
        self._movement_history.append(move_distance)
        if len(self._movement_history) > self.MAX_MOVEMENT_HISTORY:
            self._movement_history.pop(0)
        
        if self._movement_history:
            # 計算統計值
            mean_movement = np.mean(self._movement_history)
            std_movement = np.std(self._movement_history)
            total_movement = sum(self._movement_history)
            
            # 計算獎勵和懲罰
            movement_threshold = mean_movement + std_movement
            movement_reward = min(1.0, move_distance / movement_threshold)
            movement_penalty = -1.0 / (1.0 + np.exp((total_movement - 5.0) / 2.0))
        else:
            movement_reward = 0.0
            movement_penalty = 0.0

        # 更新位置記錄
        self._last_position = [x1, self._last_position[1], z1]
        return movement_reward, movement_penalty

    def calculate_posture_penalty(self, rotation: Dict[str, float]) -> float:
        """計算姿態相關的懲罰。
        
        使用rotation 歐拉角度計算姿態偏差，
        當偏差超過閾值時給予懲罰。
        
        參數:
            rotation: 包含x,z軸旋轉角度的字典
            
        回傳:
            float: 姿態懲罰值 (0.0-1.0)
        """
        # 計算x和z軸的旋轉偏差
        x_rotation = rotation.get('x', 0.0) % 360.0
        z_rotation = rotation.get('z', 0.0) % 360.0
        
        x_deviation = min(x_rotation, 360.0 - x_rotation)
        z_deviation = min(z_rotation, 360.0 - z_rotation)
        
        # 計算最大偏差並給予懲罰
        max_deviation = max(x_deviation, z_deviation)
        if max_deviation > self.ROTATION_THRESHOLD:
            normalized_penalty = (max_deviation - self.ROTATION_THRESHOLD) / (180.0 - self.ROTATION_THRESHOLD)
            return min(1.0, normalized_penalty)
        return 0.0

    def calculate_touch_rewards(self, target_positions: List[Dict[str, float]], 
                              crawler_position: Dict[str, float], reward_data: Dict[str, Any]) -> Tuple[float, float]:
        """計算碰觸目標的獎勵。
        
        根據智能體與目標的距離計算碰觸獎勵，
        並提供持續性獎勵以鼓勵維持接觸。
        
        參數:
            target_positions: 目標位置列表
            crawler_position: 智能體位置
            
        回傳:
            Tuple[float, float]: (碰觸獎勵, 持續獎勵)
        """
        touch_reward = 0.0
        continuous_reward = 0.0

        # 檢查每個目標
        for i, target_pos in enumerate(target_positions):
            target_id = f"target_{i}"
            
            # 初始化目標狀態
            if target_id not in self._target_reward_states:
                self._target_reward_states[target_id] = True
                
            # 檢查是否可獲得獎勵
            if self._target_reward_states[target_id]:
                # 計算水平面距離
                dx = target_pos['x'] - crawler_position['x']
                dz = target_pos['z'] - crawler_position['z']
                distance = np.sqrt(dx**2 + dz**2)
                # 判定碰觸
                if bool(reward_data['is_colliding']) is True or distance < self.TOUCH_THRESHOLD:
                    self._target_reward_states[target_id] = False
                    touch_reward = 1.0
                    self._continuous_touch_reward = 100.0
                    break
        
        # 計算持續獎勵
        if self._continuous_touch_reward > 0:
            remaining_targets = sum(1 for v in self._target_reward_states.values() if v)
            decay_rate = 0.95 + (remaining_targets * 0.01)
            
            continuous_reward = self._continuous_touch_reward
            self._continuous_touch_reward *= decay_rate
            
            if self._continuous_touch_reward < 1.0:
                self._continuous_touch_reward = 0.0

        return touch_reward, continuous_reward

    def find_nearest_target(self, reward_data: Dict[str, Any]) -> Tuple[NDArray, Dict[str, float]]:
        """找出最近的目標。
        
        在水平面(x-z平面)上計算距離，找出最近的目標。
        
        參數:
            reward_data: 包含智能體和目標位置資訊的字典
            
        回傳:
            Tuple[NDArray, Dict[str, float]]: (最近目標位置, 目標螢幕座標)
        """
        # 提取智能體位置
        crawler_position = np.array([
            reward_data['position']['x'],
            reward_data['position']['z']
        ])
        
        # 提取所有目標位置
        targets = reward_data['targets']
        target_positions = [
            np.array([t['position']['x'], t['position']['z']])
            for t in targets
        ]
        
        # 找出最近目標
        distances = [np.linalg.norm(crawler_position - pos) for pos in target_positions]
        nearest_idx = np.argmin(distances)
        
        return target_positions[nearest_idx], targets[nearest_idx]['screenPosition']

    def get_reward(self, detection_results: List[Any], reward_data: Dict[str, Any]) -> Tuple[float, List[float]]:
        """計算總體獎勵值。
        
        整合所有子獎勵函數的結果，並根據權重計算最終獎勵。
        
        參數:
            detection_results: YOLO偵測結果
            reward_data: 環境狀態資料
            
        回傳:
            Tuple[float, List[float]]: (總獎勵值, 各子獎勵列表)
        """
        # 提取必要資料
        crawler_pos = reward_data['position']
        target_pos, target_view_pos = self.find_nearest_target(reward_data)
        rotation = reward_data['rotation']
        target_positions = [target['position'] for target in reward_data['targets']]
        
        # 計算各項獎勵
        detection_reward = self.calculate_detection_reward(detection_results)
        approach_reward, leave_penalty = self.calculate_distance_rewards(crawler_pos, target_pos)
        in_view_reward, center_approach_reward, center_leave_penalty, lost_view_penalty = \
            self.calculate_view_rewards(target_view_pos)
        movement_reward, movement_penalty = self.calculate_movement_rewards(crawler_pos)
        posture_penalty = self.calculate_posture_penalty(rotation)
        touch_reward, continuous_reward = self.calculate_touch_rewards(target_positions, crawler_pos, reward_data)

        # 套用權重計算最終獎勵
        reward_list = [
            detection_reward * self.DETECTION_WEIGHT,           # 人像偵測獎勵
            approach_reward * self.APPROACH_WEIGHT,            # 接近目標獎勵
            leave_penalty * self.LEAVE_PENALTY_WEIGHT,         # 遠離目標懲罰
            in_view_reward * self.IN_VIEW_WEIGHT,             # 目標在視野內獎勵
            center_approach_reward * self.CENTER_APPROACH_WEIGHT,  # 目標接近視野中心獎勵
            center_leave_penalty * self.CENTER_LEAVE_WEIGHT,   # 目標遠離視野中心懲罰
            lost_view_penalty * self.LOST_VIEW_WEIGHT,        # 失去目標視野懲罰
            movement_reward * self.MOVEMENT_WEIGHT,            # 移動獎勵
            movement_penalty * self.MOVEMENT_PENALTY_WEIGHT,   # 移動距離懲罰
            posture_penalty * self.POSTURE_PENALTY_WEIGHT,    # 姿態偏差懲罰
            touch_reward * self.TOUCH_WEIGHT,                 # 碰觸目標獎勵
            continuous_reward * self.CONTINUOUS_TOUCH_WEIGHT   # 持續碰觸獎勵
        ]
        
        return np.sum(reward_list), reward_list
