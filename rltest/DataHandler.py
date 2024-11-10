import os
import h5py
import numpy as np
import threading
import queue
import torch

class DataHandler:
    def __init__(self, base_dir="train_logs", feature_save_interval=10):
        self.base_dir = base_dir
        self.feature_save_interval = feature_save_interval
        
        # 創建子目錄
        self.env_dir = os.path.join(base_dir, "env_data")
        self.feature_dir = os.path.join(base_dir, "feature_data")
        os.makedirs(self.env_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        
        self.env_file = None
        self.feature_file = None
        self.current_max_steps = 0
        self.current_max_feature_steps = 0
        self.resize_step = 1
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.latest_epoch = self._get_latest_epoch()

    def _get_latest_epoch(self):
        h5_files = [f for f in os.listdir(self.env_dir) if f.endswith(".h5")]
        if not h5_files:
            return 0
        epochs = [int(f.split('ep')[1].split('_')[0]) for f in h5_files if 'ep' in f]
        return max(epochs) if epochs else 0

    def create_epoch_file(self, epoch=None):
        if epoch is None:
            epoch = self.latest_epoch + 1

        self.latest_epoch = epoch
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        feature_file_path = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
        self.close_epoch_file()

        self.current_max_steps = 0
        self.current_max_feature_steps = 0
        self.env_file = h5py.File(env_file_path, 'w')
        self.feature_file = h5py.File(feature_file_path, 'w')

        # 創建環境數據集
        self.create_env_datasets()
        # 創建特徵數據集
        self.create_feature_datasets()

        print(f"環境數據檔案已創建: {env_file_path}")
        print(f"特徵數據檔案已創建: {feature_file_path}")
        
        self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
        self.writer_thread.start()

    def create_env_datasets(self):
        chunk_size = (100, 384, 640, 3)
        
        self.env_datasets = {
            'obs': self.env_file.create_dataset(
                'obs', (0, 384, 640, 3), 
                maxshape=(None, 384, 640, 3), 
                dtype=np.uint8, 
                chunks=chunk_size
            ),
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
                maxshape=(None, 100, 4), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_scores': self.env_file.create_dataset(
                'yolo_scores', (0, 10), 
                maxshape=(None, 100), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_classes': self.env_file.create_dataset(
                'yolo_classes', (0, 10), 
                maxshape=(None, 100), 
                dtype=np.int32, 
                chunks=True
            )
        }

    def create_feature_datasets(self):
        # 注意：特徵數據集的初始大小設為0，但會根據實際需要擴展
        self.feature_datasets = {
            'input': self.feature_file.create_dataset(
                'layer_input', (0, 3, 224, 224),
                maxshape=(None, 3, 224, 224),
                dtype=np.float32,
                chunks=True
            ),
            'conv1_output': self.feature_file.create_dataset(
                'layer_conv1', (0, 64, 112, 112),
                maxshape=(None, 64, 112, 112),
                dtype=np.float32,
                chunks=True
            ),
            'final_residual_output': self.feature_file.create_dataset(
                'layer_final_residual', (0, 512, 7, 7),
                maxshape=(None, 512, 7, 7),
                dtype=np.float32,
                chunks=True
            ),
            'features_output': self.feature_file.create_dataset(
                'layer_feature', (0, 512),
                maxshape=(None, 512),
                dtype=np.float32,
                chunks=True
            ),
            'actor_output': self.feature_file.create_dataset(
                'layer_actor', (0, 9),
                maxshape=(None, 9),
                dtype=np.float32,
                chunks=True
            ),
            # 添加一個映射數據集來記錄特徵數據對應的環境step
            'step_mapping': self.feature_file.create_dataset(
                'step_mapping', (0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=True
            )
        }

    def _resize_datasets(self, step):
        """
        根據需要調整數據集大小
        """
        # 1. 調整環境數據集大小
        new_max_steps = max(step + 1, self.current_max_steps + self.resize_step)
        if new_max_steps > self.current_max_steps:
            print(f"擴展環境資料集大小到 {new_max_steps} 步數")
            for dataset in self.env_datasets.values():
                current_shape = list(dataset.shape)
                current_shape[0] = new_max_steps
                dataset.resize(tuple(current_shape))
            self.current_max_steps = new_max_steps

        # 2. 調整特徵數據集大小
        expected_feature_steps = (step + self.feature_save_interval - 1) // self.feature_save_interval
        if expected_feature_steps > self.current_max_feature_steps:
            print(f"擴展特徵資料集大小到 {expected_feature_steps} 步數")
            for dataset in self.feature_datasets.values():
                current_shape = list(dataset.shape)
                current_shape[0] = expected_feature_steps
                dataset.resize(tuple(current_shape))
            self.current_max_feature_steps = expected_feature_steps

    def save_step_data(self, step, obs, angle_degrees, reward, reward_list, 
                    origin_image, results, layer_outputs):
        """
        儲存每一步的數據，但特徵數據只每隔特定步數儲存一次
        """
        if step >= self.current_max_steps:
            self._resize_datasets(step)
        
        # 環境數據每步都保存
        env_data = {
            'obs': obs,
            'angle_degrees': angle_degrees,
            'reward': reward,
            'reward_list': reward_list,
            'origin_image': origin_image,
            'results': results
        }
        
        # 特徵數據只在指定間隔保存
        # 注意：step從1開始，所以用(step-1)來判斷
        should_save_features = ((step-1) % self.feature_save_interval) == 0
        feature_data = layer_outputs if should_save_features else None
        
        data = {
            'step': step,
            'env_data': env_data,
            'feature_data': feature_data,
            'should_save_features': should_save_features  # 新增標記
        }
        
        self.write_queue.put(data)

    def _async_writer(self):
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                data = self.write_queue.get(timeout=1)
                self._write_data_to_hdf5(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"異步寫入時發生錯誤: {e}")

    def _write_data_to_hdf5(self, data):
        """
        將數據寫入HDF5文件
        """
        try:
            step = data['step'] - 1  # 調整step從0開始
            
            # 1. 寫入環境數據
            env_data = data['env_data']
            if env_data['obs'] is not None:
                self.env_datasets['obs'][step] = env_data['obs']
            if env_data['angle_degrees'] is not None:
                self.env_datasets['angle_degrees'][step] = env_data['angle_degrees']
            if env_data['reward'] is not None:
                self.env_datasets['reward'][step] = env_data['reward']
            if env_data['reward_list'] is not None:
                self.env_datasets['reward_list'][step] = env_data['reward_list']
            if env_data['origin_image'] is not None:
                self.env_datasets['origin_image'][step] = env_data['origin_image']

            # 2. 處理YOLO結果
            results = env_data['results']
            if results is not None:
                try:
                    if hasattr(results, 'boxes'):
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

                        self.env_datasets['yolo_boxes'][step] = boxes
                        self.env_datasets['yolo_scores'][step] = scores
                        self.env_datasets['yolo_classes'][step] = classes.astype(np.int32)
                except Exception as e:
                    print(f"處理YOLO結果時發生錯誤: {e}")
                    self.env_datasets['yolo_boxes'][step] = np.zeros((10, 4), dtype=np.float32)
                    self.env_datasets['yolo_scores'][step] = np.zeros(10, dtype=np.float32)
                    self.env_datasets['yolo_classes'][step] = np.zeros(10, dtype=np.int32)

            # 3. 寫入特徵數據（只在指定間隔寫入）
            if data['should_save_features'] and data['feature_data'] is not None:
                feature_step = step // self.feature_save_interval
                feature_data = data['feature_data']
                
                for name, tensor in feature_data.items():
                    if tensor is not None and name in self.feature_datasets:
                        try:
                            if isinstance(tensor, torch.Tensor):
                                tensor = tensor.cpu().numpy()
                            print('寫入特徵資料',tensor.shape)
                            self.feature_datasets[name][feature_step] = tensor
                        except Exception as e:
                            print(f"處理層 {name} 的輸出時發生錯誤: {e}")
                
                # 記錄特徵數據對應的環境step
                self.feature_datasets['step_mapping'][feature_step] = step
                print(f"特徵數據已保存 (step {step}, feature_step {feature_step})")

            print(f"步數 {step + 1} 的數據已保存")

        except Exception as e:
            print(f"寫入數據時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def _pad_or_trim_array(self, arr, target_shape):
        arr = np.array(arr)
        if len(target_shape) == 1:
            if len(arr) > target_shape[0]:
                return arr[:target_shape[0]]
            elif len(arr) < target_shape[0]:
                padding = [(0, target_shape[0] - len(arr))]
                return np.pad(arr, padding, mode='constant')
            return arr
        if len(arr) > target_shape[0]:
            arr = arr[:target_shape[0]]
        elif len(arr) < target_shape[0]:
            padding = [(0, target_shape[0] - len(arr)), (0, 0)]
            arr = np.pad(arr, padding, mode='constant')
        return arr

    def close_epoch_file(self):
        if hasattr(self, 'layer_outputs'):
            delattr(self, 'layer_outputs')
        
        self.stop_event.set()
        if self.writer_thread is not None:
            self.writer_thread.join()
        
        with self.write_queue.mutex:
            self.write_queue.queue.clear()
        
        if self.env_file is not None:
            self.env_file.close()
            print("環境數據檔案已關閉")
        
        if self.feature_file is not None:
            self.feature_file.close()
            print("特徵數據檔案已關閉")
        
        self.stop_event.clear()
        self.writer_thread = None