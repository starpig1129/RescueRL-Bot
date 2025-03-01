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
    def __init__(self, base_dir: str = "train_logs", feature_save_interval: int = 5, 
                 image_save_interval: int = 1, reward_save_interval: int = 1, logger=None):
        self.base_dir = base_dir
        self.feature_save_interval = feature_save_interval
        self.image_save_interval = image_save_interval
        # 設置為1以確保每一步都有圖像
        self.reward_save_interval = reward_save_interval
        self.logger = logger
        
        # 創建目錄
        self.env_dir = os.path.join(base_dir, "env_data")
        self.feature_dir = os.path.join(base_dir, "feature_data")
        os.makedirs(self.env_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 初始化文件和計數器
        self.env_file: Optional[h5py.File] = None
        self.feature_file: Optional[h5py.File] = None
        self.storage_counts = {
            'reward': 0,    # reward數據實際存儲數量
            'image': 0,     # 圖像數據實際存儲數量
            'feature': 0    # 特徵數據實際存儲數量
        }
        
        # 初始化映射表
        self.step_mappings = {
            'reward': {},   # step -> 儲存索引的映射
            'image': {},    # step -> 儲存索引的映射
            'feature': {}   # step -> 儲存索引的映射
        }
        
        # 設置異步寫入
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.latest_epoch = self._get_latest_epoch()
        
        # 初始化統計
        self.stats = {
            'total_data_saved': 0,
            'current_epoch_data': 0,
            'total_features_saved': 0,
            'current_epoch_features': 0,
            'last_save_time': 0,
            'data_save_rate': 0,
            'write_queue_size': 0,
            'disk_usage': 0
        }

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(f"[DataHandler] {message}")
        else:
            print(f"[DataHandler] {message}")

    def _log_error(self, error: Exception) -> None:
        if self.logger:
            self.logger.log_error(error)
        else:
            print(f"[DataHandler Error] {str(error)}")
            import traceback
            traceback.print_exc()

    def _get_latest_epoch(self) -> int:
        try:
            h5_files = [f for f in os.listdir(self.env_dir) if f.endswith(".h5")]
            if not h5_files:
                return 0
            epochs = [int(f.split('ep')[1].split('_')[0]) for f in h5_files if 'ep' in f]
            return max(epochs) if epochs else 0
        except Exception as e:
            self._log_error(e)
            return 0

    def create_datasets(self, data_type: str, initial_size: int = 100) -> None:
        """創建指定類型的數據集，使用初始大小"""
        if data_type == 'reward':
            self.env_datasets = {
                'angle_degrees': self.env_file.create_dataset(
                    'angle_degrees', (initial_size,), 
                    maxshape=(None,), 
                    dtype=np.float32, 
                    chunks=True
                ),
                'reward': self.env_file.create_dataset(
                    'reward', (initial_size,), 
                    maxshape=(None,), 
                    dtype=np.float32, 
                    chunks=True
                ),
                'reward_list': self.env_file.create_dataset(
                    'reward_list', (initial_size, 12), 
                    maxshape=(None, 12), 
                    dtype=np.float32, 
                    chunks=True
                ),
                'reward_step_map': self.env_file.create_dataset(
                    'reward_step_map', (initial_size,), 
                    maxshape=(None,), 
                    dtype=np.int32, 
                    chunks=True
                )
            }
            
        elif data_type == 'image':
            chunk_size = (1, 384, 640, 3)
            self.env_datasets.update({
                'origin_image': self.env_file.create_dataset(
                    'origin_image', (initial_size, 384, 640, 3), 
                    maxshape=(None, 384, 640, 3), 
                    dtype=np.uint8, 
                    chunks=chunk_size
                ),
                'top_view': self.env_file.create_dataset(
                    'top_view', (initial_size, 256, 256, 3), 
                    maxshape=(None, 256, 256, 3), 
                    dtype=np.uint8, 
                    chunks=(1, 256, 256, 3)
                ),
                'yolo_boxes': self.env_file.create_dataset(
                    'yolo_boxes', (initial_size, 10, 4), 
                    maxshape=(None, 10, 4), 
                    dtype=np.float32, 
                    chunks=True
                ),
                'yolo_scores': self.env_file.create_dataset(
                    'yolo_scores', (initial_size, 10), 
                    maxshape=(None, 10), 
                    dtype=np.float32, 
                    chunks=True
                ),
                'yolo_classes': self.env_file.create_dataset(
                    'yolo_classes', (initial_size, 10), 
                    maxshape=(None, 10), 
                    dtype=np.int32, 
                    chunks=True
                ),
                'image_step_map': self.env_file.create_dataset(
                    'image_step_map', (initial_size,), 
                    maxshape=(None,), 
                    dtype=np.int32, 
                    chunks=True
                )
            })
            
        elif data_type == 'feature':
            self.feature_datasets = {
                'input': self.feature_file.create_dataset(
                    'layer_input', (initial_size, 3, 224, 224),
                    maxshape=(None, 3, 224, 224),
                    dtype=np.float32,
                    chunks=True
                ),
                'conv1_output': self.feature_file.create_dataset(
                    'layer_conv1', (initial_size, 64, 112, 112),
                    maxshape=(None, 64, 112, 112),
                    dtype=np.float32,
                    chunks=True
                ),
                'final_residual_output': self.feature_file.create_dataset(
                    'layer_final_residual', (initial_size, 512, 7, 7),
                    maxshape=(None, 512, 7, 7),
                    dtype=np.float32,
                    chunks=True
                ),
                'features_output': self.feature_file.create_dataset(
                    'layer_feature', (initial_size, 512),
                    maxshape=(None, 512),
                    dtype=np.float32,
                    chunks=True
                ),
                'actor_output': self.feature_file.create_dataset(
                    'layer_actor', (initial_size, 3),
                    maxshape=(None, 3),
                    dtype=np.float32,
                    chunks=True
                ),
                'temporal_features': self.feature_file.create_dataset(
                    'temporal_features', (initial_size, 256),
                    maxshape=(None, 256),
                    dtype=np.float32,
                    chunks=True
                ),
                'feature_step_map': self.feature_file.create_dataset(
                    'feature_step_map', (initial_size,),
                    maxshape=(None,),
                    dtype=np.int32,
                    chunks=True
                )
            }

    def _resize_dataset(self, dataset: h5py.Dataset, new_size: int) -> None:
        """根據需要調整數據集大小"""
        current_shape = list(dataset.shape)
        current_shape[0] = new_size
        dataset.resize(tuple(current_shape))

    def _ensure_space(self, data_type: str) -> None:
        """確保有足夠的存儲空間"""
        datasets = self.feature_datasets if data_type == 'feature' else self.env_datasets
        current_count = self.storage_counts[data_type]
        
        # 檢查是否需要擴展空間
        for dataset in datasets.values():
            if current_count >= dataset.shape[0]:
                new_size = max(dataset.shape[0] * 2, current_count + 100)
                self._resize_dataset(dataset, new_size)

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

            # 重置計數器和映射
            self.storage_counts = {'reward': 0, 'image': 0, 'feature': 0}
            self.step_mappings = {'reward': {}, 'image': {}, 'feature': {}}
            
            # 創建新文件
            self.env_file = h5py.File(env_file_path, 'w')
            self.feature_file = h5py.File(feature_file_path, 'w')

            # 創建數據集
            self.create_datasets('reward', 100)
            self.create_datasets('image', 100)
            self.create_datasets('feature', 100)

            self._log_info(f"已創建世代 {epoch} 的數據文件")
            
            # 啟動異步寫入線程
            self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
            self.writer_thread.start()

        except Exception as e:
            self._log_error(e)
            raise

    def save_step_data(self, step: int, current_epoch: int, obs: np.ndarray, angle_degrees: float,
                      reward: float, reward_list: List[float], origin_image: np.ndarray,
                      results: Any, layer_outputs: Dict[str, np.ndarray],
                      top_view_image: Optional[np.ndarray] = None) -> None:
        """保存一個步驟的數據"""
        try:
            # 驗證基本參數
            if current_epoch < 0:
                self._log_error(Exception(f"無效的世代編號: {current_epoch}"))
                return

            if step < 0:
                self._log_error(Exception(f"無效的步驟編號: {step}"))
                return
            # 準備環境數據
            env_data = {
                'angle_degrees': angle_degrees,
                'reward': reward,
                'reward_list': reward_list,
                'origin_image': origin_image,
                'top_view': top_view_image,
                'results': results
            }
            
            # 驗證基本數據
            if origin_image is None or reward is None:
                self._log_error(Exception(f"步數 {step} 缺少必要數據"))
                return

            # 根據世代判斷是否需要保存
            should_save_rewards = (current_epoch % self.reward_save_interval) == 0
            should_save_images = (current_epoch % self.image_save_interval) == 0
            should_save_features = (step % self.feature_save_interval) == 0 and layer_outputs is not None

            data = {
                'step': step,
                'env_data': env_data,
                'feature_data': layer_outputs if layer_outputs is not None else None,
                'should_save_rewards': should_save_rewards,
                'should_save_images': should_save_images,
                'should_save_features': should_save_features
            }

            # 基本驗證
            if should_save_images and (origin_image is None or np.all(origin_image == 0)):
                self._log_error(Exception(f"步數 {step} 的圖像數據無效"))
                return

            # 檢查是否需要保存任何數據
            if not (should_save_rewards or should_save_images):
                return

            self.write_queue.put(data)
            
        except Exception as e:
            self._log_error(e)
            raise

            
    def _write_data_to_hdf5(self, data: Dict[str, Any]) -> None:
        try:
            should_save_rewards = data['should_save_rewards']
            should_save_images = data['should_save_images']
            step = data['step']
            env_data = data['env_data']
            
            
            
            # 使用同一個索引確保數據同步
            idx = self.storage_counts['reward']

            
            # 確保兩個數據集都有足夠空間
            self._ensure_space('reward')
            self._ensure_space('image')

            if self.storage_counts['reward'] != self.storage_counts['image']:
                self._log_error(Exception(f"數據不同步: reward={self.storage_counts['reward']}, image={self.storage_counts['image']}"))
                return
            self.env_datasets['angle_degrees'][idx] = env_data['angle_degrees']
            self.env_datasets['reward'][idx] = env_data['reward']
            self.env_datasets['reward_list'][idx] = env_data['reward_list']
            self.env_datasets['reward_step_map'][idx] = step
            
            self.step_mappings['reward'][step] = idx
            self.storage_counts['reward'] += 1
            
            # 始終保存圖像相關數據以確保連續性
            if data['should_save_images'] and env_data['origin_image'] is not None:
                # 檢查圖像質量
                if np.all(env_data['origin_image'] == 0):
                    self._log_error(Exception(f"步數 {step} 的圖像全黑"))
                    return

                # 保存圖像數據
                self._ensure_space('image')
                idx = self.storage_counts['image']
                
                self.env_datasets['origin_image'][idx] = env_data['origin_image']
                self.env_datasets['image_step_map'][idx] = step
                
                # 如果有頂部視角影像，則一併保存
                if env_data['top_view'] is not None:
                    self.env_datasets['top_view'][idx] = env_data['top_view']
                
                # 處理YOLO結果
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

                        boxes = self._pad_or_trim_array(boxes, (10, 4))
                        scores = self._pad_or_trim_array(scores, (10,))
                        classes = self._pad_or_trim_array(classes, (10,))

                        self.env_datasets['yolo_boxes'][idx] = boxes
                        self.env_datasets['yolo_scores'][idx] = scores
                        self.env_datasets['yolo_classes'][idx] = classes.astype(np.int32)
                    except Exception as e:
                        self._log_error(e)
                        self.env_datasets['yolo_boxes'][idx] = np.zeros((10, 4), dtype=np.float32)
                        self.env_datasets['yolo_scores'][idx] = np.zeros(10, dtype=np.float32)
                        self.env_datasets['yolo_classes'][idx] = np.zeros(10, dtype=np.int32)
                
                self.step_mappings['image'][step] = idx
                self.storage_counts['image'] += 1
            else:
                # 確保圖像計數與獎勵計數同步
                self.storage_counts['image'] += 1
                # 不記錄日誌以減少輸出
                return
            
            # 保存特徵數據
            if data['should_save_features'] and data['feature_data'] is not None:
                self._ensure_space('feature')
                idx = self.storage_counts['feature']
                
                feature_data = data['feature_data']
                for name, tensor in feature_data.items():
                    if tensor is not None and name in self.feature_datasets:
                        if isinstance(tensor, torch.Tensor):
                            tensor = tensor.cpu().numpy()
                        # 只取第一個時序的數據
                        if len(tensor.shape) > 2 and tensor.shape[0] == 50:
                            tensor = tensor[0]
                        self.feature_datasets[name][idx] = tensor
                
                self.feature_datasets['feature_step_map'][idx] = step
                self.step_mappings['feature'][step] = idx
                self.storage_counts['feature'] += 1
            
            self._update_stats('data')
            
        except Exception as e:
            self._log_error(e)
            import traceback
            traceback.print_exc()

    def _async_writer(self) -> None:
        """異步寫入線程"""
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                data = self.write_queue.get(timeout=1)
                self._write_data_to_hdf5(data)
            except queue.Empty:
                continue
            except Exception as e:
                self._log_error(e)

    def _update_stats(self, data_type: str) -> None:
        current_time = time.time()
        
        if data_type == 'data':
            self.stats['total_data_saved'] += 1
            self.stats['current_epoch_data'] += 1
        
        if self.stats['last_save_time'] > 0:
            time_diff = current_time - self.stats['last_save_time']
            if time_diff > 0:
                current_rate = 1.0 / time_diff
                alpha = 0.1
                self.stats['data_save_rate'] = (alpha * current_rate + 
                    (1 - alpha) * self.stats['data_save_rate'])
        
        self.stats['last_save_time'] = current_time
        self.stats['write_queue_size'] = self.write_queue.qsize()
        
        if self.env_file is not None:
            try:
                self.stats['disk_usage'] = os.path.getsize(self.env_file.filename) / (1024 * 1024)
            except:
                pass
                
            # 檢查數據連續性
            if (self.storage_counts['image'] < self.storage_counts['reward'] or
                abs(self.storage_counts['image'] - self.storage_counts['reward']) > 1):
                self._log_error(Exception(
                    f"數據不連續: 圖像={self.storage_counts['image']}, 獎勵={self.storage_counts['reward']}"))
        
        if self.logger:
            self.logger.update_data_handler_stats(self.stats)

    def _pad_or_trim_array(self, arr: np.ndarray, target_shape: Union[tuple, int]) -> np.ndarray:
        arr = np.array(arr)
        if isinstance(target_shape, int) or len(target_shape) == 1:
            target_len = target_shape if isinstance(target_shape, int) else target_shape[0]
            if len(arr) > target_len:
                return arr[:target_len]
            elif len(arr) < target_len:
                padding = [(0, target_len - len(arr))]
                return np.pad(arr, padding, mode='constant')
            return arr
            
        if len(arr) > target_shape[0]:
            arr = arr[:target_shape[0]]
        elif len(arr) < target_shape[0]:
            padding = [(0, target_shape[0] - len(arr))] + [(0, 0)] * (len(target_shape) - 1)
            arr = np.pad(arr, padding, mode='constant')
        return arr

    def close_epoch_file(self) -> None:
        """安全地關閉當前的時期文件"""
        try:
            # 1. 首先停止寫入線程
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            
            # 2. 等待寫入線程完成
            if hasattr(self, 'writer_thread') and self.writer_thread is not None:
                self.writer_thread.join(timeout=5)
            
            # 3. 處理剩餘的數據
            try:
                while not self.write_queue.empty():
                    data = self.write_queue.get_nowait()
                    self._write_data_to_hdf5(data)
            except queue.Empty:
                pass
            except Exception as e:
                self._log_error(e)

            # 4. 記錄最終統計
            if self.env_file is not None:
                final_counts = {
                    'reward': self.storage_counts['reward'],
                    'image': self.storage_counts['image']
                }
                self._log_info(f"環境數據最終統計: {final_counts}")

            if self.feature_file is not None:
                final_feature_count = self.storage_counts['feature']
                self._log_info(f"特徵數據最終數量: {final_feature_count}")

            # 5. 關閉並清理檔案
            try:
                if self.env_file is not None:
                    self.env_file.close()
            except:
                pass
            finally:
                self.env_file = None

            try:
                if self.feature_file is not None:
                    self.feature_file.close()
            except:
                pass
            finally:
                self.feature_file = None
            
            self.writer_thread = None
            self.stop_event.clear()
            
            self._log_info(
                f"世代 {self.latest_epoch} 統計:\n"
                f"- 獎勵數據量: {self.storage_counts['reward']}\n"
                f"- 圖像數據量: {self.storage_counts['image']}\n"
                f"- 特徵數據量: {self.storage_counts['feature']}\n"
                f"- 平均保存速率: {self.stats['data_save_rate']:.2f} 步/秒\n"
                f"- 磁碟使用量: {self.stats['disk_usage']:.2f} MB"
            )

        except Exception as e:
            self._log_error(e)
            raise

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def __del__(self):
        self.close_epoch_file()
