import cv2
import numpy as np
import time
import argparse
from DataReader import DataReader

class EnhancedReplayViewer:
    def __init__(self, base_dir="train_logs", fps=30):
        """
        初始化增強版回放查看器
        
        Args:
            base_dir (str): 數據目錄路徑
            fps (int): 播放的每秒幀數
        """
        self.data_reader = DataReader(base_dir=base_dir)
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # 創建主要視窗
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Observation', cv2.WINDOW_NORMAL)
        
        # 創建特徵圖視窗
        cv2.namedWindow('Input Features', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Conv1 Features', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Final Residual', cv2.WINDOW_NORMAL)
        
        # 設置視窗大小和位置
        cv2.resizeWindow('Original Image', 640, 384)
        cv2.resizeWindow('Observation', 640, 384)
        cv2.resizeWindow('Input Features', 320, 320)
        cv2.resizeWindow('Conv1 Features', 320, 320)
        cv2.resizeWindow('Final Residual', 320, 320)
        
        # 移動視窗到適當位置
        cv2.moveWindow('Original Image', 0, 0)
        cv2.moveWindow('Observation', 650, 0)
        cv2.moveWindow('Input Features', 0, 450)
        cv2.moveWindow('Conv1 Features', 330, 450)
        cv2.moveWindow('Final Residual', 660, 450)

    def normalize_feature_map(self, feature_map):
        """
        標準化特徵圖以便視覺化
        """
        if feature_map is None:
            return None
            
        # 移除批次維度如果存在
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
            
        # 計算每個通道的平均特徵圖
        if len(feature_map.shape) == 3:
            feature_map = np.mean(feature_map, axis=0)
            
        # 標準化到 0-255 範圍
        if feature_map.max() != feature_map.min():
            feature_map = ((feature_map - feature_map.min()) * 255 / 
                         (feature_map.max() - feature_map.min()))
        else:
            feature_map = np.zeros_like(feature_map)
            
        return feature_map.astype(np.uint8)

    def create_feature_grid(self, features, grid_size=(8, 8), max_channels=64):
        """
        將多通道特徵圖排列成網格
        """
        if features is None or len(features.shape) < 3:
            return None
            
        # 移除批次維度如果存在
        if len(features.shape) == 4:
            features = features[0]
            
        channels = min(features.shape[0], max_channels)
        h, w = features.shape[1:3]
        
        # 計算網格維度
        grid_h, grid_w = grid_size
        cell_h, cell_w = h, w
        
        # 創建空白網格
        grid = np.zeros((cell_h * grid_h, cell_w * grid_w), dtype=np.uint8)
        
        # 填充網格
        for idx in range(min(channels, grid_h * grid_w)):
            i, j = idx // grid_w, idx % grid_w
            feature = features[idx]
            
            # 標準化單個特徵圖
            normalized = self.normalize_feature_map(feature)
            
            # 填充到網格中
            grid[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = normalized
            
        return grid

    def visualize_features(self, layer_outputs):
        """
        視覺化不同層的特徵圖
        """
        feature_maps = {}
        
        # 處理輸入特徵
        if 'input' in layer_outputs:
            input_features = layer_outputs['input']
            input_viz = self.normalize_feature_map(input_features)
            if input_viz is not None:
                feature_maps['input'] = cv2.applyColorMap(input_viz, cv2.COLORMAP_VIRIDIS)
        
        # 處理 Conv1 輸出
        if 'conv1_output' in layer_outputs:
            conv1_features = layer_outputs['conv1_output']
            conv1_grid = self.create_feature_grid(conv1_features, (8, 8))
            if conv1_grid is not None:
                feature_maps['conv1'] = cv2.applyColorMap(conv1_grid, cv2.COLORMAP_VIRIDIS)
        
        # 處理最終殘差層輸出
        if 'final_residual_output' in layer_outputs:
            residual_features = layer_outputs['final_residual_output']
            residual_grid = self.create_feature_grid(residual_features, (8, 8))
            if residual_grid is not None:
                feature_maps['residual'] = cv2.applyColorMap(residual_grid, cv2.COLORMAP_VIRIDIS)
        
        return feature_maps

    def display_stats(self, frame, angle, reward, reward_list):
        """在影像上顯示統計資訊"""
        # 基本資訊
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reward: {reward:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 獎勵列表詳細資訊
        if reward_list is not None:
            reward_names = [
                "Person Detection", "Distance+", "Distance-", "InView",
                "ViewDist+", "ViewDist-", "LostView", "Movement+",
                "Movement-", "UpsideDown", "Touch", "Continuous"
            ]
            y_offset = 110
            for name, value in zip(reward_names, reward_list):
                if value != 0:  # 只顯示非零值
                    color = (0, 255, 0) if value > 0 else (0, 0, 255)
                    cv2.putText(frame, f"{name}: {value:.2f}", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20

    def display_detection(self, frame, boxes, scores, classes):
        """顯示檢測結果"""
        if boxes is None or len(boxes) == 0:
            return

        for box, score, cls in zip(boxes, scores, classes):
            if score > 0:  # 只顯示有效的檢測
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def play_range(self, epoch, start_step, end_step):
        """播放指定範圍的記錄"""
        print(f"正在讀取世代 {epoch} 的數據...")
        data = self.data_reader.load_range_data(epoch, slice(start_step, end_step))
        
        if data is None:
            print("無法讀取數據")
            return
            
        print(f"成功讀取數據，開始播放...")
        print("按 'q' 退出，空白鍵暫停/繼續，左右方向鍵調整播放速度")
        
        paused = False
        frame_idx = 0
        total_frames = len(data['obs'])
        
        while frame_idx < total_frames:
            if not paused:
                # 獲取當前幀的所有數據
                obs = data['obs'][frame_idx]
                origin_image = data['origin_image'][frame_idx]
                angle = data['angle_degrees'][frame_idx]
                reward = data['reward'][frame_idx]
                reward_list = data['reward_list'][frame_idx]
                
                # 複製影像以繪製資訊
                obs_display = obs.copy()
                origin_display = origin_image.copy()
                
                # 顯示統計資訊
                self.display_stats(origin_display, angle, reward, reward_list)
                
                # 顯示檢測結果
                if 'yolo_boxes' in data:
                    self.display_detection(
                        obs_display,
                        data['yolo_boxes'][frame_idx],
                        data['yolo_scores'][frame_idx],
                        data['yolo_classes'][frame_idx]
                    )
                
                # 讀取並視覺化特徵圖
                feature_maps = self.visualize_features({
                    'input': data.get('layer_input', [None])[frame_idx],
                    'conv1_output': data.get('layer_conv1', [None])[frame_idx],
                    'final_residual_output': data.get('layer_final_residual', [None])[frame_idx]
                })
                
                # 顯示所有視窗
                cv2.imshow('Original Image', origin_display)
                cv2.imshow('Observation', obs_display)
                
                if 'input' in feature_maps:
                    cv2.imshow('Input Features', feature_maps['input'])
                if 'conv1' in feature_maps:
                    cv2.imshow('Conv1 Features', feature_maps['conv1'])
                if 'residual' in feature_maps:
                    cv2.imshow('Final Residual', feature_maps['residual'])
                
                frame_idx += 1
            
            # 等待按鍵輸入
            key = cv2.waitKey(int(self.frame_delay * 1000))
            
            if key == ord('q'):  # 按 q 退出
                break
            elif key == ord(' '):  # 按空白鍵暫停/繼續
                paused = not paused
            elif key == 81:  # 左方向鍵，降低播放速度
                self.fps = max(1, self.fps - 5)
                self.frame_delay = 1.0 / self.fps
                print(f"播放速度: {self.fps} FPS")
            elif key == 83:  # 右方向鍵，提高播放速度
                self.fps += 5
                self.frame_delay = 1.0 / self.fps
                print(f"播放速度: {self.fps} FPS")
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='增強版訓練記錄回放器')
    parser.add_argument('--dir', type=str, default='train_logs',
                        help='數據目錄路徑')
    parser.add_argument('--epoch', type=int, required=True,
                        help='要播放的世代')
    parser.add_argument('--start', type=int, default=0,
                        help='起始步數')
    parser.add_argument('--end', type=int, default=None,
                        help='結束步數')
    parser.add_argument('--fps', type=int, default=30,
                        help='播放的每秒幀數')
    
    args = parser.parse_args()
    
    # 創建查看器並播放
    viewer = EnhancedReplayViewer(base_dir=args.dir, fps=args.fps)
    
    # 如果沒有指定結束步數，獲取最大步數
    if args.end is None:
        max_steps = viewer.data_reader.get_max_steps(args.epoch)
        if max_steps is None:
            print(f"無法獲取世代 {args.epoch} 的最大步數")
            return
        args.end = max_steps
    
    viewer.play_range(args.epoch, args.start, args.end)

if __name__ == "__main__":
    main()