import os
import h5py
import numpy as np
import threading
import queue
import torch
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

class DataHandler:
    """
    數據處理器類別，用於處理和保存訓練過程中的數據
    
    主要功能：
    - 異步保存環境觀察數據
    - 保存神經網絡特徵數據
    - 管理訓練世代(epoch)的數據文件
    - 處理YOLO目標檢測結果
    - 提供詳細的數據保存統計
    """
    
    def __init__(self, base_dir: str = "train_logs", feature_save_interval: int = 10, logger=None):
        """
        初始化數據處理器
        
        參數：
            base_dir: 數據保存的基礎目錄
            feature_save_interval: 特徵保存的間隔步數
            logger: 日誌記錄器實例
        """
        self.base_dir = base_dir
        self.feature_save_interval = feature_save_interval
        self.logger = logger
        
        # 創建環境數據和特徵數據的子目錄
        self.env_dir = os.path.join(base_dir, "env_data")
        self.feature_dir = os.path.join(base_dir, "feature_data")
        os.makedirs(self.env_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 初始化文件和計數器
        self.env_file: Optional[h5py.File] = None
        self.feature_file: Optional[h5py.File] = None
        self.current_max_steps = 0
        self.current_max_feature_steps = 0
        self.resize_step = 1000  # 每次擴展數據集的步數
        
        # 初始化資料集字典
        self.env_datasets: Dict[str, h5py.Dataset] = {}
        self.feature_datasets: Dict[str, h5py.Dataset] = {}
        
        # 設置異步寫入相關變量
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.latest_epoch = self._get_latest_epoch()
        
        # 初始化統計信息
        self.stats = {
            'total_data_saved': 0,          # 總共保存的數據量
            'current_epoch_data': 0,        # 當前世代保存的數據量
            'total_features_saved': 0,      # 總共保存的特徵數據量
            'current_epoch_features': 0,    # 當前世代保存的特徵數據量
            'last_save_time': 0,           # 上次保存數據的時間
            'data_save_rate': 0,           # 數據保存速率
            'write_queue_size': 0,         # 寫入隊列大小
            'disk_usage': 0,               # 磁碟使用量
            'last_resize_time': 0,         # 上次調整數據集大小的時間
            'total_resize_count': 0        # 總共調整數據集大小的次數
        }

    def _log_info(self, message: str) -> None:
        """記錄信息"""
        if self.logger:
            self.logger.log_info(f"[DataHandler] {message}")
        else:
            print(f"[DataHandler] {message}")

    def _log_error(self, error: Exception) -> None:
        """記錄錯誤"""
        if self.logger:
            self.logger.log_error(error)
        else:
            print(f"[DataHandler Error] {str(error)}")
            import traceback
            traceback.print_exc()

    def _get_latest_epoch(self) -> int:
        """獲取目前最新的世代號碼"""
        try:
            h5_files = [f for f in os.listdir(self.env_dir) if f.endswith(".h5")]
            if not h5_files:
                return 0
            epochs = [int(f.split('ep')[1].split('_')[0]) for f in h5_files if 'ep' in f]
            return max(epochs) if epochs else 0
        except Exception as e:
            self._log_error(e)
            return 0

    def _update_stats(self, data_type: str) -> None:
        """更新數據統計信息"""
        current_time = time.time()
        
        if data_type == 'data':
            self.stats['total_data_saved'] += 1
            self.stats['current_epoch_data'] += 1
        elif data_type == 'feature':
            self.stats['total_features_saved'] += 1
            self.stats['current_epoch_features'] += 1
        
        # 計算保存速率（使用移動平均）
        if self.stats['last_save_time'] > 0:
            time_diff = current_time - self.stats['last_save_time']
            if time_diff > 0:
                current_rate = 1.0 / time_diff
                # 使用移動平均平滑速率
                alpha = 0.1  # 平滑因子
                self.stats['data_save_rate'] = (alpha * current_rate + 
                    (1 - alpha) * self.stats['data_save_rate'])
        
        self.stats['last_save_time'] = current_time
        self.stats['write_queue_size'] = self.write_queue.qsize()
        
        # 更新磁碟使用量
        if self.env_file is not None:
            try:
                self.stats['disk_usage'] = os.path.getsize(self.env_file.filename) / (1024 * 1024)
            except:
                pass
        
        # 立即通知logger更新統計信息
        if self.logger:
            self.logger.update_data_handler_stats(self.stats)

    def create_env_datasets(self) -> None:
        """創建環境數據相關的數據集，初始大小為0，移除obs只保留原始圖像"""
        chunk_size = (100, 384, 640, 3)  # 影像數據的分塊大小
        
        self.env_datasets = {
            'angle_degrees': self.env_file.create_dataset(
                'angle_degrees', (0,), 
                maxshape=(None,), 
                dtype=np.float32, 
                chunks=True
            ),
            'reward': self.env_file.create_dataset(
                'reward', (0,), 
                maxshape=(None,), 
                dtype=np.float32, 
                chunks=True
            ),
            'reward_list': self.env_file.create_dataset(
                'reward_list', (0, 12), 
                maxshape=(None, 12), 
                dtype=np.float32, 
                chunks=True
            ),
            'origin_image': self.env_file.create_dataset(
                'origin_image', (0, 384, 640, 3), 
                maxshape=(None, 384, 640, 3), 
                dtype=np.uint8, 
                chunks=chunk_size
            ),
            'yolo_boxes': self.env_file.create_dataset(
                'yolo_boxes', (0, 10, 4), 
                maxshape=(None, 10, 4), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_scores': self.env_file.create_dataset(
                'yolo_scores', (0, 10), 
                maxshape=(None, 10), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_classes': self.env_file.create_dataset(
                'yolo_classes', (0, 10), 
                maxshape=(None, 10), 
                dtype=np.int32, 
                chunks=True
            )
        }
        self.current_max_steps = 0

    def create_feature_datasets(self) -> None:
        """創建神經網絡特徵相關的數據集，初始大小為0，只保留第一個時序的影像內容"""
        self.feature_datasets = {
            'input': self.feature_file.create_dataset(
                'layer_input', (0, 3, 224, 224),  # 增加時序維度
                maxshape=(None, 3, 224, 224),
                dtype=np.float32,
                chunks=True
            ),
            'conv1_output': self.feature_file.create_dataset(
                'layer_conv1', (0, 64, 112, 112),  # 增加時序維度
                maxshape=(None, 64, 112, 112),
                dtype=np.float32,
                chunks=True
            ),
            'final_residual_output': self.feature_file.create_dataset(
                'layer_final_residual', (0, 512, 7, 7),  # 增加時序維度
                maxshape=(None, 512, 7, 7),
                dtype=np.float32,
                chunks=True
            ),
            'features_output': self.feature_file.create_dataset(
                'layer_feature', (0, 512),  # 移除時序維度，只保存特徵向量
                maxshape=(None, 512),
                dtype=np.float32,
                chunks=True
            ),
            'actor_output': self.feature_file.create_dataset(
                'layer_actor', (0, 3),  # 動作輸出
                maxshape=(None, 3),
                dtype=np.float32,
                chunks=True
            ),
            'temporal_features': self.feature_file.create_dataset(
                'temporal_features', (0, 256),  # LSTM輸出的時序特徵
                maxshape=(None, 256),
                dtype=np.float32,
                chunks=True
            ),
            'step_mapping': self.feature_file.create_dataset(
                'step_mapping', (0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=True
            )
        }
        self.current_max_feature_steps = 0

    def _resize_datasets(self, step: int) -> None:
        """
        精確調整數據集大小
        
        參數：
            step: 當前步數
        """
        try:
            # 根據當前步數精確調整環境數據集大小
            target_size = step + 1  # 確保有足夠空間存儲當前步數的數據
            if target_size > self.current_max_steps:
                for dataset in self.env_datasets.values():
                    current_shape = list(dataset.shape)
                    current_shape[0] = target_size
                    dataset.resize(tuple(current_shape))
                self.current_max_steps = target_size

            # 根據需要精確調整特徵數據集大小
            feature_step = step // self.feature_save_interval
            if feature_step >= self.current_max_feature_steps:
                target_feature_size = feature_step + 1
                
                for dataset in self.feature_datasets.values():
                    current_shape = list(dataset.shape)
                    current_shape[0] = target_feature_size
                    dataset.resize(tuple(current_shape))
                
                self.current_max_feature_steps = target_feature_size

            # 更新統計信息
            self.stats['last_resize_time'] = time.time()
            self.stats['total_resize_count'] += 1

        except Exception as e:
            self._log_error(e)
            raise

    def create_epoch_file(self, epoch: Optional[int] = None) -> None:
        """創建新的世代數據文件"""
        try:
            if epoch is None:
                epoch = self.latest_epoch + 1

            self.latest_epoch = epoch
            env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
            feature_file_path = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
            
            # 關閉之前的文件
            self.close_epoch_file()

            # 重置計數器和統計信息
            self.current_max_steps = 0
            self.current_max_feature_steps = 0
            self.stats['current_epoch_data'] = 0
            self.stats['current_epoch_features'] = 0
            
            # 創建新的HDF5文件
            self.env_file = h5py.File(env_file_path, 'w')
            self.feature_file = h5py.File(feature_file_path, 'w')

            # 創建數據集
            self.create_env_datasets()
            self.create_feature_datasets()

            self._log_info(f"已創建世代 {epoch} 的數據文件")
            
            # 啟動異步寫入線程
            self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
            self.writer_thread.start()

        except Exception as e:
            self._log_error(e)
            raise

    def _async_writer(self) -> None:
        """異步寫入線程的主函數"""
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                data = self.write_queue.get(timeout=1)
                self._write_data_to_hdf5(data)
            except queue.Empty:
                continue
            except Exception as e:
                self._log_error(e)

    def save_step_data(self, step: int, obs: np.ndarray, angle_degrees: float, 
                      reward: float, reward_list: List[float], origin_image: np.ndarray, 
                      results: Any, layer_outputs: Dict[str, np.ndarray]) -> None:
        """保存一個步驟的所有相關數據"""
        try:
            if step >= self.current_max_steps:
                self._resize_datasets(step)
            
            # 準備環境數據，不保存obs
            env_data = {
                'angle_degrees': angle_degrees,
                'reward': reward,
                'reward_list': reward_list,
                'origin_image': origin_image,
                'results': results
            }
            
            # 確定是否需要保存特徵數據
            should_save_features = ((step-1) % self.feature_save_interval) == 0
            
            # 如果需要保存特徵，只保存第一個時序的數據
            feature_data = None
            if should_save_features and layer_outputs is not None:
                feature_data = {}
                for name, tensor in layer_outputs.items():
                    if tensor is not None and name in self.feature_datasets:
                        if isinstance(tensor, torch.Tensor):
                            tensor = tensor.cpu().numpy()
                        # 只取第一個時序的數據
                        if len(tensor.shape) > 2 and tensor.shape[0] == 50:  # 有時序維度
                            tensor = tensor[0]  # 只取第一個時序
                        feature_data[name] = tensor
            
            # 打包數據並放入隊列
            data = {
                'step': step,
                'env_data': env_data,
                'feature_data': feature_data,
                'should_save_features': should_save_features
            }
            
            self.write_queue.put(data)
            
            # 更新統計信息
            self._update_stats('data')
            if should_save_features:
                self._update_stats('feature')

        except Exception as e:
            self._log_error(e)
            raise

    def _write_data_to_hdf5(self, data: Dict[str, Any]) -> None:
        """
        將數據寫入HDF5文件
        
        參數：
            data: 包含所有需要寫入的數據的字典
        """
        try:
            # 調整步數從0開始
            step = data['step'] - 1
            
            # 寫入環境數據
            env_data = data['env_data']
            for key in ['angle_degrees', 'reward', 'reward_list', 'origin_image']:
                if env_data.get(key) is not None:
                    self.env_datasets[key][step] = env_data[key]

            # 處理YOLO檢測結果
            results = env_data['results']
            if results is not None and hasattr(results, 'boxes'):
                try:
                    if torch.is_tensor(results.boxes.xyxy):
                        boxes = results.boxes.xyxy.cpu().numpy()
                        scores = results.boxes.conf.cpu().numpy()
                        classes = results.boxes.cls.cpu().numpy()
                    else:
                        boxes = results.boxes.xyxy
                        scores = results.boxes.conf
                        classes = results.boxes.cls

                    # 調整數組大小
                    boxes = self._pad_or_trim_array(boxes, (10, 4))
                    scores = self._pad_or_trim_array(scores, (10,))
                    classes = self._pad_or_trim_array(classes, (10,))

                    # 保存數據
                    self.env_datasets['yolo_boxes'][step] = boxes
                    self.env_datasets['yolo_scores'][step] = scores
                    self.env_datasets['yolo_classes'][step] = classes.astype(np.int32)
                except Exception as e:
                    self._log_error(e)
                    # 發生錯誤時使用零值填充
                    self.env_datasets['yolo_boxes'][step] = np.zeros((10, 4), dtype=np.float32)
                    self.env_datasets['yolo_scores'][step] = np.zeros(10, dtype=np.float32)
                    self.env_datasets['yolo_classes'][step] = np.zeros(10, dtype=np.int32)

            # 寫入特徵數據（只在指定間隔寫入）
            if data['should_save_features'] and data['feature_data'] is not None:
                feature_step = step // self.feature_save_interval
                feature_data = data['feature_data']
                
                # 遍歷所有特徵層
                for name, tensor in feature_data.items():
                    if tensor is not None and name in self.feature_datasets:
                        try:
                            self.feature_datasets[name][feature_step] = tensor
                        except Exception as e:
                            self._log_error(e)
                
                # 記錄特徵數據對應的環境step
                self.feature_datasets['step_mapping'][feature_step] = step

        except Exception as e:
            self._log_error(e)
            import traceback
            traceback.print_exc()

    def _pad_or_trim_array(self, arr: np.ndarray, target_shape: Union[tuple, int]) -> np.ndarray:
        """
        調整數組大小：截斷過長的數組或填充過短的數組
        
        參數：
            arr: 輸入數組
            target_shape: 目標形狀
            
        返回：
            調整後的數組
        """
        arr = np.array(arr)
        # 處理一維數組
        if isinstance(target_shape, int) or len(target_shape) == 1:
            target_len = target_shape if isinstance(target_shape, int) else target_shape[0]
            if len(arr) > target_len:
                return arr[:target_len]
            elif len(arr) < target_len:
                padding = [(0, target_len - len(arr))]
                return np.pad(arr, padding, mode='constant')
            return arr
            
        # 處理多維數組
        if len(arr) > target_shape[0]:
            arr = arr[:target_shape[0]]
        elif len(arr) < target_shape[0]:
            padding = [(0, target_shape[0] - len(arr))] + [(0, 0)] * (len(target_shape) - 1)
            arr = np.pad(arr, padding, mode='constant')
        return arr

    def close_epoch_file(self) -> None:
        """關閉當前世代的數據文件並清理資源"""
        try:
            # 確保所有隊列中的數據都被處理
            while not self.write_queue.empty():
                try:
                    data = self.write_queue.get_nowait()
                    self._write_data_to_hdf5(data)
                except queue.Empty:
                    break
                except Exception as e:
                    self._log_error(e)

            # 停止異步寫入線程
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            
            if hasattr(self, 'writer_thread') and self.writer_thread is not None:
                self.writer_thread.join(timeout=5)
                if self.writer_thread.is_alive():
                    self._log_info("等待寫入線程完成...")
                    self.writer_thread.join()
            
            if self.env_file is not None:
                try:
                    # 記錄最終使用的數據量
                    final_env_steps = self.current_max_steps
                    self._log_info(f"環境數據最終步數: {final_env_steps}")
                    
                    # 確保文件仍然打開再進行操作
                    if self.env_file and hasattr(self.env_file, 'id'):
                        self.env_file.flush()
                        self.env_file.close()
                        self._log_info("環境數據文件已關閉")
                except Exception as e:
                    self._log_error(e)
            
            if self.feature_file is not None:
                try:
                    # 記錄最終使用的特徵數據量
                    final_feature_steps = self.current_max_feature_steps
                    self._log_info(f"特徵數據最終步數: {final_feature_steps}")
                    
                    # 確保文件仍然打開再進行操作
                    if self.feature_file and hasattr(self.feature_file, 'id'):
                        self.feature_file.flush()
                        self.feature_file.close()
                        self._log_info("特徵數據文件已關閉")
                except Exception as e:
                    self._log_error(e)
            
            # 重置所有標誌和計數器
            self.writer_thread = None
            self.stop_event.clear()
            self.current_max_steps = 0
            self.current_max_feature_steps = 0

            # 記錄最終統計信息
            self._log_info(
                f"世代 {self.latest_epoch} 統計:\n"
                f"- 總數據量: {self.stats['current_epoch_data']}\n"
                f"- 特徵數據量: {self.stats['current_epoch_features']}\n"
                f"- 平均保存速率: {self.stats['data_save_rate']:.2f} 步/秒\n"
                f"- 磁碟使用量: {self.stats['disk_usage']:.2f} MB"
            )

        except Exception as e:
            self._log_error(e)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """獲取當前統計信息"""
        return self.stats.copy()

    def __del__(self):
        """析構函數，確保資源被正確釋放"""
        self.close_epoch_file()
