import cv2
import numpy as np
import time
import argparse
import os
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
        cv2.namedWindow('Top View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Rewards', cv2.WINDOW_NORMAL)
        
        # 創建特徵圖視窗
        cv2.namedWindow('Input Features', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Conv1 Features', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Final Residual', cv2.WINDOW_NORMAL)
        
        # 設置視窗大小和位置
        cv2.resizeWindow('Original Image', 640, 384)
        cv2.resizeWindow('Observation', 640, 384)
        cv2.resizeWindow('Top View', 512, 512)  # 增加Top View的顯示大小
        cv2.resizeWindow('Rewards', 400, 400)
        cv2.resizeWindow('Input Features', 320, 320)
        cv2.resizeWindow('Conv1 Features', 320, 320)
        cv2.resizeWindow('Final Residual', 320, 320)
        
        # 移動視窗到適當位置
        cv2.moveWindow('Original Image', 0, 0)
        cv2.moveWindow('Observation', 650, 0)
        cv2.moveWindow('Top View', 1300, 0)
        cv2.moveWindow('Rewards', 0, 450)
        cv2.moveWindow('Input Features', 410, 450)
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
        frame = frame.copy()
        cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Reward: {reward:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def create_reward_display(self, angle, reward, reward_list):
        """創建獎勵顯示視窗"""
        # 創建一個白色背景的圖像
        reward_display = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # 顯示基本信息
        cv2.putText(reward_display, f"Angle: {angle:.1f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(reward_display, f"Total Reward: {reward:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 繪製分隔線
        cv2.line(reward_display, (0, 70), (400, 70), (200, 200, 200), 2)
        cv2.putText(reward_display, "Reward Components:", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 獎勵列表詳細資訊
        if reward_list is not None:
            reward_names = [
                "Person Detection",  # 人物偵測
                "Distance+",         # 距離接近
                "Distance-",         # 距離遠離
                "InView",            # 視野內
                "ViewDist+",         # 視野距離改善
                "ViewDist-",         # 視野距離惡化
                "LostView",          # 失去視野
                "Movement+",         # 移動接近
                "Movement-",         # 移動遠離
                "UpsideDown",        # 顛倒
                "Touch",             # 接觸目標
                "Continuous"         # 持續接觸
            ]
            
            y_offset = 110
            max_value = max(abs(v) for v in reward_list) if any(reward_list) else 1.0
            bar_width = 200  # 進度條寬度
            
            for name, value in zip(reward_names, reward_list):
                if value != 0:  # 只顯示非零值
                    # 顯示獎勵名稱
                    cv2.putText(reward_display, f"{name}:", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # 繪製進度條
                    bar_length = int(abs(value) / max_value * bar_width)
                    bar_color = (0, 200, 0) if value > 0 else (0, 0, 200)
                    
                    # 進度條起始位置
                    start_x = 150

                    cv2.rectangle(reward_display, (start_x, y_offset-10), (start_x + bar_length, y_offset-2), bar_color, -1)
                    
                    # 顯示數值
                    # 根據值的正負調整文字位置
                    if value > 0:
                        cv2.putText(reward_display, f"{value:.2f}", (start_x + bar_width + 10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        # 負值時，將文字放在進度條上方而不是左側
                        # 進一步調整位置，確保文字不被進度條覆蓋
                        # 將負值文字放在與正值相同的位置（進度條右側），確保一致性和可讀性
                        cv2.putText(reward_display, f"{value:.2f}", (start_x + bar_width + 10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    y_offset += 25
        
        return reward_display

    def get_video_writer(self, output_path, frame_size):
        """
        創建視頻寫入器，嘗試不同的編碼器
        """
        codecs = [
            ('avc1', '.mp4'),
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi')
        ]
        
        writer = None
        for codec, ext in codecs:
            try:
                file_path = output_path.rsplit('.', 1)[0] + ext
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(file_path, fourcc, self.fps, frame_size)
                if writer.isOpened():
                    print(f"使用 {codec} 編碼器創建視頻文件")
                    return writer, file_path
                writer.release()
            except Exception as e:
                if writer is not None:
                    writer.release()
                print(f"編碼器 {codec} 不可用: {str(e)}")
                continue
        return None, None

    def synthesize_obs(self, img, boxes, scores, classes):
        """
        合成觀察數據，將原始圖像和 YOLO 檢測結果整合
        
        Args:
            img: 原始圖像
            boxes: YOLO 檢測框 [N, 4]
            scores: 檢測分數 [N]
            classes: 類別標籤 [N]
            
        Returns:
            合成後的觀察圖像
        """
        # 確保輸入圖像是正確的格式
        if img is None or not isinstance(img, np.ndarray):
            return np.zeros((384, 640, 3), dtype=np.uint8)
            
        # 複製輸入圖像以避免修改原始數據
        obs = img.copy()
        
        # 如果圖像是灰度圖，轉換為 RGB
        if len(obs.shape) == 2:
            obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2BGR)
            
        return obs


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

    def create_combined_frame(self, frames_dict):
        """創建組合影像用於視頻保存"""
        # 定義最大寬度，避免影像過寬
        MAX_WIDTH = 1920
        
        # 獲取所有影像並添加標題
        frames = []
        keys = []
        for key, frame in frames_dict.items():
            if frame is not None:
                # 添加標題
                titled_frame = np.ones((30 + frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                titled_frame[30:, :, :] = frame
                cv2.putText(titled_frame, key, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                frames.append(titled_frame)
                keys.append(key)
        
        if not frames:
            return None
        
        # 確保所有影像都有相同的通道數
        for i, frame in enumerate(frames):
            if len(frame.shape) == 2:  # 灰度圖
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:  # 單通道
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        
        # 使用網格布局而不是水平拼接
        # 根據影像數量決定網格大小
        n_frames = len(frames)
        if n_frames <= 3:
            grid_cols = n_frames  # 如果只有1-3個影像，每個一列
            grid_rows = (n_frames + grid_cols - 1) // grid_cols
        elif n_frames <= 6:
            grid_cols = 2  # 減少為2列，讓每個影像有更多空間
            grid_rows = (n_frames + grid_cols - 1) // grid_cols
        else:
            grid_cols = 2  # 減少為2列
            grid_rows = (n_frames + grid_cols - 1) // grid_cols  # 向上取整
        
        # 計算每個影像的目標大小
        target_width = min(MAX_WIDTH // grid_cols - 20, 580)  # 進一步減小寬度，確保有更多間距
        
        # 根據影像類型調整大小
        resized_frames = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            key = keys[i] if i < len(keys) else ""
            
            # 特徵圖需要更大的顯示空間
            if "Features" in key or "Residual" in key:
                # 特徵圖使用固定大小，確保能清晰顯示
                target_h = 350  # 稍微減小特徵圖大小
                target_w = 350
            else:
                # 其他影像保持寬高比
                target_h = int(h * target_width / w)
                target_w = target_width
            
            resized = cv2.resize(frame, (target_w, target_h))
            resized_frames.append(resized)
        
        # 重新排序frames_dict，將特徵圖放在一起
        # 這樣在網格布局中它們會被放在同一行
        ordered_frames = []
        ordered_keys = []
        
        # 先添加非特徵圖
        for i, key in enumerate(keys):
            if not ("Features" in key or "Residual" in key):
                ordered_frames.append(resized_frames[i])
                ordered_keys.append(key)
        
        # 再添加特徵圖
        for i, key in enumerate(keys):
            if "Features" in key or "Residual" in key:
                ordered_frames.append(resized_frames[i])
                ordered_keys.append(key)
        
        # 使用重新排序後的影像
        resized_frames = ordered_frames
        keys = ordered_keys
        
        # 找出每行的最大高度
        row_heights = []
        for row in range(grid_rows):
            start_idx = row * grid_cols
            end_idx = min(start_idx + grid_cols, n_frames)
            max_height = max([resized_frames[i].shape[0] for i in range(start_idx, end_idx)])
            row_heights.append(max_height)
        
        # 創建畫布
        canvas_width = grid_cols * target_width
        canvas_height = sum(row_heights)
        # 增加畫布寬度，為影像之間添加更多間距
        canvas = np.ones((canvas_height, canvas_width + (grid_cols-1) * 20, 3), dtype=np.uint8) * 255
        
        # 放置影像
        y_offset = 0
        for row in range(grid_rows):
            x_offset = 0
            start_idx = row * grid_cols
            end_idx = min(start_idx + grid_cols, n_frames)
            
            for i in range(start_idx, end_idx):
                frame = resized_frames[i]
                h, w = frame.shape[:2]
                
                # 垂直居中放置在當前行
                y_pos = y_offset + (row_heights[row] - h) // 2
                canvas[y_pos:y_pos+h, x_offset:x_offset+w] = frame
                
                # 添加分隔線
                if x_offset > 0:
                    cv2.line(canvas, (x_offset-1, y_offset), (x_offset-1, y_offset+row_heights[row]), (200, 200, 200), 2)
                
                # 增加影像之間的間距
                x_offset += target_width+ 20  # 添加20像素的間距
            
            # 添加水平分隔線
            if row < grid_rows - 1:
                cv2.line(canvas, (0, y_offset+row_heights[row]), (canvas_width, y_offset+row_heights[row]), (200, 200, 200), 2)
            
            y_offset += row_heights[row]
            
        try:
            return canvas
        except Exception as e:
            print(f"創建組合影像時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果上述方法失敗，嘗試簡單的拼接
            try:
                # 確保所有影像高度相同
                max_height = max(frame.shape[0] for frame in frames)
                return np.hstack([cv2.resize(f, (int(f.shape[1] * max_height / f.shape[0]), max_height)) if f.shape[0] != max_height else f for f in frames])
            except:
                return frames[0]  # 如果所有方法都失敗，至少返回第一個影像

    def play_range(self, epoch, start_step, end_step, save_video=False):
        """播放指定範圍的記錄"""
        print(f"正在讀取世代 {epoch} 的數據...")
        data = self.data_reader.load_range_data(epoch, slice(start_step, end_step, 1))
        
        if data is None:
            print(f"無法讀取世代 {epoch} 的數據")
            return
            
        # 檢查必要的數據是否存在
        required_keys = ['origin_image', 'reward', 'angle_degrees', 'reward_list']
        for key in required_keys:
            if key not in data:
                print(f"數據缺少必要的部分: {key}")
                return
                
        # 過濾有效數據
        valid_indices = []
        for i, img in enumerate(data['origin_image']):
            if np.any(img != 0):  # 檢查非零圖像
                try:
                    valid_indices.append(i)
                except Exception as e:
                    print(f"處理索引 {i} 時發生錯誤: {e}")
        total_frames = len(valid_indices)
        print(f"成功讀取數據，開始播放...")
        print(f"找到 {total_frames} 個有效幀")
        if total_frames == 0:
            return
        print("按 'q' 退出，空白鍵暫停/繼續，左右方向鍵調整播放速度")
        
        paused = False
        frame_idx = 0

        # 生成observation數據
        obs_list = []
        for i in valid_indices:
            if i < len(data['origin_image']):
                img = data['origin_image'][i]
                obs = self.synthesize_obs(img,
                    data.get('yolo_boxes', np.zeros((len(data['origin_image']), 10, 4)))[i],
                    data.get('yolo_scores', np.zeros((len(data['origin_image']), 10)))[i],
                    data.get('yolo_classes', np.zeros((len(data['origin_image']), 10)))[i])
                obs_list.append(obs)
        data['obs'] = np.array(obs_list)
                
        # 設置視頻保存
        video_writer = None
        if save_video:
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'replay_ep{epoch}_steps{start_step}-{end_step}.mp4')

            
            # 我們將在第一幀生成後再創建視頻寫入器
            video_writer = None
            if video_writer is None:
                print("無法創建視頻文件，將只進行顯示")
        
        while frame_idx < total_frames and (video_writer is None or video_writer.isOpened()):
            if not paused:
                # 獲取當前幀的所有數據
                current_idx = valid_indices[frame_idx]
                obs = data['obs'][frame_idx]
                origin_image = data['origin_image'][current_idx]
                if origin_image is None or not np.any(origin_image):
                    frame_idx += 1
                    continue
                    
                angle = data['angle_degrees'][current_idx]
                reward = data['reward'][current_idx]
                reward_list = data['reward_list'][current_idx]
                
                # 複製影像以繪製資訊
                obs_display = obs.copy()
                origin_display = origin_image.copy()
                
                # 顯示統計資訊
                self.display_stats(origin_display, angle, reward, reward_list)

                # 創建獎勵顯示
                reward_display = self.create_reward_display(angle, reward, reward_list)
                
                # 顯示檢測結果
                if 'yolo_boxes' in data:
                    self.display_detection(
                        obs_display,
                        data['yolo_boxes'][current_idx],
                        data['yolo_scores'][current_idx],
                        data['yolo_classes'][current_idx]
                    )
                
                # 讀取並視覺化特徵圖
                feature_maps = self.visualize_features({
                    'input': data.get('layer_input', [None])[current_idx],
                    'conv1_output': data.get('layer_conv1', [None])[current_idx],
                    'final_residual_output': data.get('layer_final_residual', [None])[current_idx]
                })
                
                # 顯示所有視窗
                cv2.imshow('Original Image', origin_display)
                # 顯示頂視圖
                cv2.imshow('Observation', obs_display)
                if 'top_view' in data and data['top_view'][current_idx] is not None:
                    top_view = data['top_view'][current_idx]

                    # 使用線性插值方法將Top View放大到512x512
                    resized_top_view = cv2.resize(
                        top_view, 
                        (512, 512), 
                        interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('Top View', resized_top_view)
                
                # 顯示獎勵視窗
                cv2.imshow('Rewards', reward_display)

                # 如果需要保存視頻，寫入當前幀
                if save_video:
                    try:
                        # 創建組合影像
                        frames_dict = {
                            'Original': origin_display,
                            'Observation': obs_display,
                            'Rewards': reward_display
                        }
                        
                        if 'top_view' in data and data['top_view'][current_idx] is not None:
                            frames_dict['Top View'] = data['top_view'][current_idx]
                        
                        # 單獨處理特徵圖，確保它們放在一起
                        feature_frames = {}
                        if 'input' in feature_maps and feature_maps['input'] is not None:
                            feature_frames['Input Features'] = feature_maps['input']
                        if 'conv1' in feature_maps and feature_maps['conv1'] is not None:
                            feature_frames['Conv1 Features'] = feature_maps['conv1']
                        if 'residual' in feature_maps and feature_maps['residual'] is not None:
                            feature_frames['Final Residual'] = feature_maps['residual']
                        
                        # 將特徵圖添加到frames_dict的最後
                        for key, frame in feature_frames.items():
                            frames_dict[key] = frame
                            
                        combined_frame = self.create_combined_frame(frames_dict)
                        
                        # 如果是第一幀，創建視頻寫入器
                        if video_writer is None and combined_frame is not None:
                            height, width = combined_frame.shape[:2]
                            video_writer, output_path = self.get_video_writer(output_path, (width, height))
                            
                        if video_writer is not None and combined_frame is not None:
                            video_writer.write(combined_frame)
                    except Exception as e:
                        print(f"寫入視頻時發生錯誤: {e}")
                
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

        
        # 清理資源
        if video_writer is not None:
            video_writer.release()
            cv2.destroyAllWindows()  # 確保清理所有視窗
            if os.path.exists(output_path):
                print(f"\n視頻已保存至: {output_path}")
            else:
                print("\n視頻保存失敗")
        
        cv2.waitKey(1)  # 確保所有視窗都有時間更新
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
    parser.add_argument('--save-video', action='store_true',
                        help='是否保存為視頻文件')
    
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
    
    viewer.play_range(args.epoch, args.start, args.end, args.save_video)

if __name__ == "__main__":
#python rltest/test/replay_viewer.py --dir E:/train_log0118/train_log --epoch 50 --start 0 --end 100 --save-video --fps 30
    main()