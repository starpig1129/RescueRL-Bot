import os
import h5py
import numpy as np

class DataReader:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir

    def get_max_steps(self, epoch):
        """
        讀取指定世代的最大步數
        :param epoch: 世代號碼
        :return: 最大步數 (int)
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        with h5py.File(file_path, 'r') as hdf5_file:
            if 'obs' not in hdf5_file:
                print(f"資料集 'obs' 在 {file_path} 中不存在")
                return None
            
            # 獲取 obs 資料集的 shape 的第一個維度作為最大步數
            max_steps = hdf5_file['obs'].shape[0]
            return max_steps

    def load_range_data(self, epoch, slice_obj):
        """
        使用 NumPy 樣式的切片方式讀取指定世代和步數範圍的資料
        :param epoch: 世代號碼
        :param slice_obj: NumPy 樣式的切片物件 (可以是切片或是步數範圍)
        :return: 讀取到的資料 (dict 格式，對應每個 dataset)
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        with h5py.File(file_path, 'r') as hdf5_file:
            # 確保資料集存在
            required_datasets = ['obs', 'angle_degrees', 'reward', 'reward_list', 'origin_image', 'yolo_boxes', 'yolo_scores', 'yolo_classes']
            for dataset in required_datasets:
                if dataset not in hdf5_file:
                    print(f"資料集 {dataset} 在 {file_path} 中不存在")
                    return None
            
            # 使用切片 (slice) 來讀取範圍內的資料
            data = {
                'obs': hdf5_file['obs'][slice_obj],
                'angle_degrees': hdf5_file['angle_degrees'][slice_obj],
                'reward': hdf5_file['reward'][slice_obj],
                'reward_list': hdf5_file['reward_list'][slice_obj],
                'origin_image': hdf5_file['origin_image'][slice_obj],
                'yolo_boxes': hdf5_file['yolo_boxes'][slice_obj],
                'yolo_scores': hdf5_file['yolo_scores'][slice_obj],
                'yolo_classes': hdf5_file['yolo_classes'][slice_obj],
            }
            
            print(f"成功使用切片讀取世代 {epoch} 的資料")
            return data
# 使用範例
data_reader = DataReader(base_dir="train_logs")
epoch = 2

# 取得指定世代的最大步數
max_steps = data_reader.get_max_steps(epoch)
if max_steps is not None:
    print(f"世代 {epoch} 的最大步數為: {max_steps}")
else:
    print("異常")
data_range = data_reader.load_range_data(epoch, slice(5, 1000, 100))

if data_range:
    print("成功讀取範圍資料:")
    print("觀察空間 shape:", data_range['obs'].shape)
    print("動作角度:", data_range['angle_degrees'])
    print("獎勵:", data_range['reward'])
else:
    print("無範圍資料")
