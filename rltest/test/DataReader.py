import os
import h5py
import numpy as np
import cv2
from typing import Optional, Dict, Any, List, Union

class DataReader:
    def __init__(self, base_dir="train_logs"):
        self.base_dir = base_dir
        self.env_dir = os.path.join(base_dir, "env_data")
        self.feature_dir = os.path.join(base_dir, "feature_data")
        
        # YOLO類別顏色映射
        self.colors = {
            0: (0, 255, 0),    # 人 - 綠色
        }
    
    def _draw_boxes(self, image: np.ndarray, boxes: np.ndarray, 
                   scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """在圖像上繪製YOLO檢測框"""
        img = image.copy()
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.75:  # 只繪製置信度大於0.75的檢測結果
                x1, y1, x2, y2 = box.astype(int)
                color = self.colors.get(cls, (128, 128, 128))  # 未知類別使用灰色
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"Class {cls}: {score:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
    
    def synthesize_obs(self, origin_image: np.ndarray, boxes: np.ndarray, 
                      scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """從原始圖像和YOLO檢測結果合成obs"""
        return self._draw_boxes(origin_image, boxes, scores, classes)
    
    def get_max_steps(self, epoch):
        """讀取指定世代的最大步數"""
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        
        if not os.path.exists(env_file_path):
            print(f"檔案 {env_file_path} 不存在")
            return None
        
        try:
            with h5py.File(env_file_path, 'r') as env_file:
                # 使用reward_step_map獲取最大步數
                if 'reward_step_map' in env_file:
                    return int(np.max(env_file['reward_step_map'][:])) + 1
                else:
                    print(f"找不到reward_step_map數據集")
                    return None
        except Exception as e:
            print(f"讀取最大步數時發生錯誤: {e}")
            return None
    
    def _find_nearest_index(self, step: int, step_map: np.ndarray, max_distance: int = 10) -> Optional[int]:
        """找到最接近的已儲存步數的索引
        
        Args:
            step: 目標步數
            step_map: 步數映射表
            max_distance: 允許的最大差距步數
        
        Returns:
            最近的索引，如果距離超過max_distance則返回None
        """
        if len(step_map) == 0:
            return None
            
        distances = np.abs(step_map - step)
        nearest_idx = distances.argmin()
        min_distance = distances[nearest_idx]
        
        # 如果最近的數據點距離太遠，返回None而不是使用該數據點
        if min_distance > max_distance:
            return None
        
        return nearest_idx
    
    def load_range_data(self, epoch, slice_obj):
        """使用NumPy樣式的切片方式讀取指定世代和步數範圍的資料"""
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        feature_file_path = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
        
        if not os.path.exists(env_file_path):
            print(f"環境數據檔案 {env_file_path} 不存在")
            return None
            
        try:
            # 準備返回的數據字典
            aligned_data = {}
            
            # 打開環境數據檔案
            with h5py.File(env_file_path, 'r') as env_file:
                # 獲取實際步數範圍
                if isinstance(slice_obj, slice):
                    start = slice_obj.start if slice_obj.start is not None else 0
                    stop = slice_obj.stop if slice_obj.stop is not None else self.get_max_steps(epoch)
                    step_range = range(start, stop, slice_obj.step or 1)
                else:
                    step_range = [slice_obj]
                
                # 讀取step映射表和設置各數據類型的最大允許距離
                reward_steps = env_file['reward_step_map'][:]
                image_steps = env_file['image_step_map'][:]
                
                reward_max_distance = 1   # reward要求精確對應
                image_max_distance = 5    # 圖像可以接受5步以內的差距
                feature_max_distance = 10  # 特徵可以接受10步以內的差距
                
                # 初始化數據數組
                aligned_data['reward'] = np.zeros(len(step_range), dtype=np.float32)
                aligned_data['reward_list'] = np.zeros((len(step_range), 12), dtype=np.float32)
                aligned_data['angle_degrees'] = np.zeros(len(step_range), dtype=np.float32)
                aligned_data['origin_image'] = np.zeros((len(step_range), 384, 640, 3), dtype=np.uint8)
                aligned_data['yolo_boxes'] = np.zeros((len(step_range), 10, 4), dtype=np.float32)
                aligned_data['yolo_scores'] = np.zeros((len(step_range), 10), dtype=np.float32)
                aligned_data['yolo_classes'] = np.zeros((len(step_range), 10), dtype=np.int32)
                
                # 讀取並填充數據
                for i, step in enumerate(step_range):
                    # 找到最近的reward數據（要求精確匹配）
                    reward_idx = self._find_nearest_index(step, reward_steps, reward_max_distance)
                    if reward_idx is not None:
                        aligned_data['reward'][i] = env_file['reward'][reward_idx]
                        aligned_data['reward_list'][i] = env_file['reward_list'][reward_idx]
                        aligned_data['angle_degrees'][i] = env_file['angle_degrees'][reward_idx]
                    
                    # 找到最近的圖像數據（允許一定範圍內的差距）
                    image_idx = self._find_nearest_index(step, image_steps, image_max_distance)
                    if image_idx is not None:
                        aligned_data['origin_image'][i] = env_file['origin_image'][image_idx]
                        aligned_data['yolo_boxes'][i] = env_file['yolo_boxes'][image_idx]
                        aligned_data['yolo_scores'][i] = env_file['yolo_scores'][image_idx]
                        aligned_data['yolo_classes'][i] = env_file['yolo_classes'][image_idx]
                
                # 合成obs數據
                aligned_data['obs'] = np.array([
                    self.synthesize_obs(
                        aligned_data['origin_image'][i],
                        aligned_data['yolo_boxes'][i],
                        aligned_data['yolo_scores'][i],
                        aligned_data['yolo_classes'][i]
                    )
                    for i in range(len(step_range))
                ])
            
            # 如果存在特徵文件，讀取特徵數據
            if os.path.exists(feature_file_path):
                with h5py.File(feature_file_path, 'r') as feature_file:
                    feature_steps = feature_file['feature_step_map'][:]
                    
                    # 初始化特徵數據數組
                    feature_datasets = [
                        'layer_input', 'layer_conv1', 'layer_final_residual',
                        'layer_feature', 'layer_actor', 'temporal_features'
                    ]
                    
                    for dataset in feature_datasets:
                        if dataset in feature_file:
                            shape = (len(step_range),) + feature_file[dataset].shape[1:]
                            aligned_data[dataset] = np.zeros(shape, dtype=feature_file[dataset].dtype)
                            
                            # 填充特徵數據
                            for i, step in enumerate(step_range):
                                feature_idx = self._find_nearest_index(step, feature_steps, feature_max_distance)
                                if feature_idx is not None:
                                    aligned_data[dataset][i] = feature_file[dataset][feature_idx]
            
            return aligned_data
                
        except Exception as e:
            print(f"讀取資料時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_epochs(self):
        """獲取目錄中所有可用的世代號碼"""
        try:
            h5_files = [f for f in os.listdir(self.env_dir) if f.endswith('.h5')]
            epochs = []
            for f in h5_files:
                try:
                    epoch = int(f.split('ep')[1].split('_')[0])
                    feature_file = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
                    if os.path.exists(feature_file):
                        epochs.append(epoch)
                except:
                    continue
            return sorted(epochs)
        except Exception as e:
            print(f"獲取世代列表時發生錯誤: {e}")
            return []
    
    def get_file_info(self, epoch):
        """獲取指定世代數據文件的詳細信息"""
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        feature_file_path = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
        
        if not os.path.exists(env_file_path):
            print(f"環境數據檔案 {env_file_path} 不存在")
            return None
            
        if not os.path.exists(feature_file_path):
            print(f"特徵數據檔案 {feature_file_path} 不存在")
            return None
        
        try:
            info = {
                'env_file_size': os.path.getsize(env_file_path) / (1024 * 1024),  # MB
                'feature_file_size': os.path.getsize(feature_file_path) / (1024 * 1024),  # MB
                'env_datasets': {},
                'feature_datasets': {},
                'storage_counts': {}
            }
            
            # 獲取環境數據統計
            with h5py.File(env_file_path, 'r') as env_file:
                info['storage_counts']['reward'] = len(env_file['reward_step_map'])
                info['storage_counts']['image'] = len(env_file['image_step_map'])
                
                for key in env_file.keys():
                    dataset = env_file[key]
                    info['env_datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size': dataset.size
                    }
            
            # 獲取特徵數據統計
            with h5py.File(feature_file_path, 'r') as feature_file:
                info['storage_counts']['feature'] = len(feature_file['feature_step_map'])
                
                for key in feature_file.keys():
                    dataset = feature_file[key]
                    info['feature_datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size': dataset.size
                    }
            
            return info
                
        except Exception as e:
            print(f"獲取檔案信息時發生錯誤: {e}")
            return None

if __name__ == "__main__":
    # 使用範例
    data_reader = DataReader(base_dir="train_logs")
    
    # 獲取所有可用的世代
    epochs = data_reader.get_all_epochs()
    print(f"可用的世代: {epochs}")
    
    if epochs:
        # 選擇第一個世代進行測試
        test_epoch = epochs[0]
        
        # 獲取檔案信息
        file_info = data_reader.get_file_info(test_epoch)
        if file_info:
            print(f"\n世代 {test_epoch} 的檔案信息:")
            print(f"環境數據檔案大小: {file_info['env_file_size']:.2f} MB")
            print(f"特徵數據檔案大小: {file_info['feature_file_size']:.2f} MB")
            print("\n儲存數量統計:")
            for data_type, count in file_info['storage_counts'].items():
                print(f"  {data_type}: {count}")
            print("\n環境數據集信息:")
            for name, info in file_info['env_datasets'].items():
                print(f"{name}: shape={info['shape']}, dtype={info['dtype']}")
            print("\n特徵數據集信息:")
            for name, info in file_info['feature_datasets'].items():
                print(f"{name}: shape={info['shape']}, dtype={info['dtype']}")
        
        # 讀取部分數據
        max_steps = data_reader.get_max_steps(test_epoch)
        if max_steps:
            print(f"\n世代 {test_epoch} 的最大步數: {max_steps}")
            
            # 讀取前20步的數據
            data_range = data_reader.load_range_data(test_epoch, slice(0, 20))
            if data_range:
                print("\n成功讀取數據範圍:")
                for key, value in data_range.items():
                    if isinstance(value, np.ndarray):
                        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
