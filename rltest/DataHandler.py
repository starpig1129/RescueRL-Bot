import os
import h5py
import numpy as np
import threading
import queue
import torch

class DataHandler:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.hdf5_file = None
        self.current_max_steps = 0
        self.resize_step = 1
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.latest_epoch = self._get_latest_epoch()

    def _get_latest_epoch(self):
        """
        檢查 base_dir 目錄中是否有現有的 HDF5 檔案，並找出最新的 epoch 數字
        """
        h5_files = [f for f in os.listdir(self.base_dir) if f.endswith(".h5")]
        if not h5_files:
            return 0  # 如果沒有檔案，則從第 0 代開始

        # 解析所有檔案名稱並找出最大的 epoch 數字
        epochs = [int(f.split('_')[1].split('.')[0]) for f in h5_files if 'epoch' in f]
        return max(epochs) if epochs else 0

    def create_epoch_file(self, epoch=None):
        if epoch is None:
            epoch = self.latest_epoch + 1

        self.latest_epoch = epoch
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        self.close_epoch_file()

        self.current_max_steps = 0
        self.hdf5_file = h5py.File(file_path, 'w')

        # 基本資料集
        self.create_basic_datasets()
        # 層輸出資料集
        self.create_layer_output_datasets()

        print(f"資料檔案已創建並啟用可擴展: {file_path}")
        self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
        self.writer_thread.start()
    
    def create_basic_datasets(self):
        chunk_size = (100, 384, 640, 3)
        
        self.datasets = {
            'obs': self.hdf5_file.create_dataset(
                'obs', (0, 384, 640, 3), 
                maxshape=(None, 384, 640, 3), 
                dtype=np.uint8, 
                chunks=chunk_size
            ),
            'angle_degrees': self.hdf5_file.create_dataset(
                'angle_degrees', (0,), 
                maxshape=(None,), 
                dtype=np.float32, 
                chunks=True
            ),
            'reward': self.hdf5_file.create_dataset(
                'reward', (0,), 
                maxshape=(None,), 
                dtype=np.float32, 
                chunks=True
            ),
            'reward_list': self.hdf5_file.create_dataset(
                'reward_list', (0, 12), 
                maxshape=(None, 12), 
                dtype=np.float32, 
                chunks=True
            ),
            'origin_image': self.hdf5_file.create_dataset(
                'origin_image', (0, 384, 640, 3), 
                maxshape=(None, 384, 640, 3), 
                dtype=np.uint8, 
                chunks=chunk_size
            ),
            'yolo_boxes': self.hdf5_file.create_dataset(
                'yolo_boxes', (0, 10, 4), 
                maxshape=(None, 100, 4), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_scores': self.hdf5_file.create_dataset(
                'yolo_scores', (0, 10), 
                maxshape=(None, 100), 
                dtype=np.float32, 
                chunks=True
            ),
            'yolo_classes': self.hdf5_file.create_dataset(
                'yolo_classes', (0, 10), 
                maxshape=(None, 100), 
                dtype=np.int32, 
                chunks=True
            )
        }

    def create_layer_output_datasets(self):
        """
        為神經網路中間層輸出創建資料集，根據實際的網絡架構設置正確的形狀
        """
        # 計算各層的輸出尺寸
        input_h, input_w = 384, 640
        
        # Conv1: stride=2, kernel=7
        conv1_h = (input_h + 2*3 - 7) // 2 + 1  # padding=3
        conv1_w = (input_w + 2*3 - 7) // 2 + 1
        
        # MaxPool: stride=2, kernel=3
        pool1_h = (conv1_h + 2*1 - 3) // 2 + 1  # padding=1
        pool1_w = (conv1_w + 2*1 - 3) // 2 + 1
        
        # 計算最終的特徵圖尺寸（經過多個stride=2的層）
        final_h = pool1_h // 8  # 經過3個stride=2的層
        final_w = pool1_w // 8
        
        self.layer_datasets = {
            'input': self.hdf5_file.create_dataset(
                'layer_input', (0, 3, input_h, input_w),
                maxshape=(None, 3, input_h, input_w),
                dtype=np.float32,
                chunks=True
            ),
            'conv1_output': self.hdf5_file.create_dataset(
                'layer_conv1', (0, 64, conv1_h, conv1_w),
                maxshape=(None, 64, conv1_h, conv1_w),
                dtype=np.float32,
                chunks=True
            ),
            'final_residual_output': self.hdf5_file.create_dataset(
                'layer_final_residual', (0, 512, final_h, final_w),
                maxshape=(None, 512, final_h, final_w),
                dtype=np.float32,
                chunks=True
            ),
            'feature_output': self.hdf5_file.create_dataset(
                'layer_feature', (0, 512),
                maxshape=(None, 512),
                dtype=np.float32,
                chunks=True
            ),
            'actor_output': self.hdf5_file.create_dataset(
                'layer_actor', (0, 9),
                maxshape=(None, 9),
                dtype=np.float32,
                chunks=True
            )
        }
    
        # 打印各層的形狀以便調試
        print("Layer output shapes:")
        for name, dataset in self.layer_datasets.items():
            print(f"{name}: {dataset.shape[1:]}")

    def _resize_datasets(self):
        new_max_steps = self.current_max_steps + self.resize_step
        print(f"擴展資料集大小到 {new_max_steps} 步數")

        # 調整基本資料集大小
        for dataset in self.datasets.values():
            current_shape = list(dataset.shape)
            current_shape[0] = new_max_steps
            dataset.resize(tuple(current_shape))

        # 調整層輸出資料集大小
        for dataset in self.layer_datasets.values():
            current_shape = list(dataset.shape)
            current_shape[0] = new_max_steps
            dataset.resize(tuple(current_shape))

        self.current_max_steps = new_max_steps

    def save_step_data(self, step, obs, angle_degrees, reward, reward_list, 
                      origin_image, results, layer_outputs=None):
        if step >= self.current_max_steps:
            self._resize_datasets()

        # 將所有資料打包成字典
        data = {
            'step': step,
            'obs': obs,
            'angle_degrees': angle_degrees,
            'reward': reward,
            'reward_list': reward_list,
            'origin_image': origin_image,
            'results': results,
            'layer_outputs': layer_outputs
        }
        
        self.write_queue.put(data)

    def _async_writer(self):
        """
        異步寫入的執行緒函數，從隊列中取出資料並進行寫入
        """
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                # 從 Queue 中取資料進行寫入
                data = self.write_queue.get(timeout=1)
                self._write_data_to_hdf5(data)
            except queue.Empty:
                continue  # 若 Queue 為空，則繼續等待
            except Exception as e:
                print(f"異步寫入時發生錯誤: {e}")

    def _write_data_to_hdf5(self, data):
        """
        將數據寫入 HDF5 文件
        """
        try:
            step = data['step'] - 1  # 調整索引，從 0 開始

            # 處理YOLO結果
            results = data['results']
            if results is not None:
                try:
                    # 檢查results的結構並安全地提取數據
                    if hasattr(results, 'boxes'):
                        # 如果是YOLO的原始輸出格式
                        if torch.is_tensor(results.boxes.xyxy):
                            boxes = results.boxes.xyxy.cpu().numpy()
                            scores = results.boxes.conf.cpu().numpy()
                            classes = results.boxes.cls.cpu().numpy()
                        else:
                            # 已經是numpy數組
                            boxes = results.boxes.xyxy
                            scores = results.boxes.conf
                            classes = results.boxes.cls
                    else:
                        # 假設results已經是包含所需數據的字典或類似結構
                        boxes = np.array(results.get('boxes', []))
                        scores = np.array(results.get('scores', []))
                        classes = np.array(results.get('classes', []))

                    # 確保數組不為空
                    if len(boxes) > 0:
                        # 填充或裁剪到固定大小
                        boxes = self._pad_or_trim_array(boxes, (10, 4))
                        scores = self._pad_or_trim_array(scores, (10,))
                        classes = self._pad_or_trim_array(classes, (10,))

                        # 寫入YOLO相關數據
                        self.datasets['yolo_boxes'][step] = boxes
                        self.datasets['yolo_scores'][step] = scores
                        self.datasets['yolo_classes'][step] = classes.astype(np.int32)
                    else:
                        # 如果沒有檢測到物體，填充零
                        self.datasets['yolo_boxes'][step] = np.zeros((10, 4), dtype=np.float32)
                        self.datasets['yolo_scores'][step] = np.zeros(10, dtype=np.float32)
                        self.datasets['yolo_classes'][step] = np.zeros(10, dtype=np.int32)

                except Exception as e:
                    print(f"處理YOLO結果時發生錯誤: {e}")
                    print(f"Results type: {type(results)}")
                    print(f"Results content: {results}")
                    # 填充零值作為後備
                    self.datasets['yolo_boxes'][step] = np.zeros((10, 4), dtype=np.float32)
                    self.datasets['yolo_scores'][step] = np.zeros(10, dtype=np.float32)
                    self.datasets['yolo_classes'][step] = np.zeros(10, dtype=np.int32)

            # 寫入其他基本數據
            if data.get('obs') is not None:
                self.datasets['obs'][step] = data['obs']
            if data.get('angle_degrees') is not None:
                self.datasets['angle_degrees'][step] = data['angle_degrees']
            if data.get('reward') is not None:
                self.datasets['reward'][step] = data['reward']
            if data.get('reward_list') is not None:
                self.datasets['reward_list'][step] = data['reward_list']
            if data.get('origin_image') is not None:
                self.datasets['origin_image'][step] = data['origin_image']

            # 處理層輸出數據
            layer_outputs = data.get('layer_outputs')
            if layer_outputs is not None:
                for name, tensor in layer_outputs.items():
                    if tensor is not None and name in self.layer_datasets:
                        try:
                            # 確保數據是numpy數組
                            if isinstance(tensor, torch.Tensor):
                                tensor = tensor.cpu().numpy()
                            self.layer_datasets[name][step] = tensor
                        except Exception as e:
                            print(f"處理層 {name} 的輸出時發生錯誤: {e}")

            print(f"步數 {step + 1} 的數據已保存")

        except Exception as e:
            print(f"寫入數據時發生錯誤: {e}")
            print(f"錯誤類型: {type(e)}")
            print(f"錯誤詳情: {str(e)}")
            import traceback
            traceback.print_exc()
    def _pad_or_trim_array(self, arr, target_shape):
        """
        填充或裁剪數組到目標形狀
        """
        arr = np.array(arr)  # 確保輸入是numpy數組
        
        # 如果是一維數組
        if len(target_shape) == 1:
            if len(arr) > target_shape[0]:
                return arr[:target_shape[0]]
            elif len(arr) < target_shape[0]:
                padding = [(0, target_shape[0] - len(arr))]
                return np.pad(arr, padding, mode='constant')
            return arr
            
        # 如果是二維數組
        if len(arr) > target_shape[0]:
            arr = arr[:target_shape[0]]
        elif len(arr) < target_shape[0]:
            padding = [(0, target_shape[0] - len(arr)), (0, 0)]
            arr = np.pad(arr, padding, mode='constant')
            
        return arr
    def _pad_or_trim_array(self, arr, target_shape):
        """
        填充或裁剪數組到目標形狀
        """
        arr = np.array(arr)  # 確保輸入是numpy數組
        
        # 如果是一維數組
        if len(target_shape) == 1:
            if len(arr) > target_shape[0]:
                return arr[:target_shape[0]]
            elif len(arr) < target_shape[0]:
                padding = [(0, target_shape[0] - len(arr))]
                return np.pad(arr, padding, mode='constant')
            return arr
            
        # 如果是二維數組
        if len(arr) > target_shape[0]:
            arr = arr[:target_shape[0]]
        elif len(arr) < target_shape[0]:
            padding = [(0, target_shape[0] - len(arr)), (0, 0)]
            arr = np.pad(arr, padding, mode='constant')
            
        return arr
    def close_epoch_file(self):
        """
        關閉當前的epoch文件並清理資源
        """
        # 清除當前的layer_outputs
        if hasattr(self, 'layer_outputs'):
            delattr(self, 'layer_outputs')
        
        # 停止寫入線程
        self.stop_event.set()
        if self.writer_thread is not None:
            self.writer_thread.join()
        
        # 清空寫入隊列
        with self.write_queue.mutex:
            self.write_queue.queue.clear()
        
        # 關閉HDF5文件
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            print("HDF5 檔案已關閉")
        
        # 重置事件和線程狀態
        self.stop_event.clear()
        self.writer_thread = None