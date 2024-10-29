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
        return max(epochs)
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
        # 為神經網路中間層輸出創建資料集
        self.layer_datasets = {
            'input': self.hdf5_file.create_dataset(
                'layer_input', (0, 3, 224, 224),
                maxshape=(None, 3, 224, 224),
                dtype=np.float32,
                chunks=True
            ),
            'conv1_output': self.hdf5_file.create_dataset(
                'layer_conv1', (0, 64, 112, 112),
                maxshape=(None, 64, 112, 112),
                dtype=np.float32,
                chunks=True
            ),
            'final_residual_output': self.hdf5_file.create_dataset(
                'layer_final_residual', (0, 512, 7, 7),
                maxshape=(None, 512, 7, 7),
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

    def save_step_data(self, step, obs, angle_degrees, reward, reward_list, 
                      origin_image, results, layer_outputs=None):
        if step >= self.current_max_steps:
            self._resize_datasets()

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

    def _async_writer(self):
        """
        異步寫入的執行緒函數，從隊列中取出資料並進行寫入
        """
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                # 從 Queue 中取資料進行寫入
                step, obs, angle_degrees, reward, reward_list, origin_image, results = self.write_queue.get(timeout=1)
                self._write_data_to_hdf5(step, obs, angle_degrees, reward, reward_list, origin_image, results)
            except queue.Empty:
                continue  # 若 Queue 為空，則繼續等待
            except Exception as e:
                print(f"異步寫入時發生錯誤: {e}")

    def _write_data_to_hdf5(self, data):
        """
        将数据写入 HDF5 文件
        """
        step = data['step'] - 1  # 调整索引，从 0 开始
        
        try:
            # 写入基本数据
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

            # 处理层输出
            layer_outputs = data.get('layer_outputs', {})
            if layer_outputs is not None:
                for name, tensor in layer_outputs.items():
                    if tensor is not None and name in self.layer_datasets:
                        try:
                            # 确保数据是numpy数组并具有正确的形状
                            if isinstance(tensor, torch.Tensor):
                                tensor = tensor.cpu().numpy()
                            self.layer_datasets[name][step] = tensor
                        except Exception as e:
                            print(f"写入层 {name} 的输出时发生错误: {e}")

            print(f"步数 {step + 1} 的数据已保存")

        except Exception as e:
            print(f"写入数据时发生错误: {e}")


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
