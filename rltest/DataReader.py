import h5py
import numpy as np
import os

class DataReader:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir

    def load_step_data(self, epoch, step):
        """
        讀取指定世代和步數的資料
        :param epoch: 世代號碼
        :param step: 步數號碼
        :return: 讀取到的資料 (dict 格式)
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        with h5py.File(file_path, 'r') as hdf5_file:
            # 檢查該步數資料是否存在
            step_group_name = f"step_{step:03d}"
            if step_group_name not in hdf5_file:
                print(f"步數 {step} 的資料在 {file_path} 中不存在")
                return None
            
            step_group = hdf5_file[step_group_name]
            
            # 讀取資料
            data = {
                'obs': np.array(step_group['obs']),
                'angle_degrees': step_group['angle_degrees'][()],
                'reward': step_group['reward'][()],
                'reward_list': np.array(step_group['reward_list']),
                'origin_image': np.array(step_group['origin_image']),
                'yolo_boxes': np.array(step_group['yolo_boxes']),
                'yolo_scores': np.array(step_group['yolo_scores']),
                'yolo_classes': np.array(step_group['yolo_classes']),
                'yolo_masks': np.array(step_group['yolo_masks']) if 'yolo_masks' in step_group else None,
                'yolo_keypoints': np.array(step_group['yolo_keypoints']) if 'yolo_keypoints' in step_group else None
            }
            
            print(f"成功讀取世代 {epoch} 步數 {step} 的資料")
            return data

# 使用範例
data_reader = DataReader(base_dir="env_data")
epoch = 1
step = 50
data = data_reader.load_step_data(epoch, step)

if data:
    print("觀察空間 shape:", data['obs'].shape)
    print("動作角度:", data['angle_degrees'])
    print("獎勵:", data['reward'])
    print("YOLO 偵測框:", data['yolo_boxes'])
    print("YOLO 置信度:", data['yolo_scores'])
    print("YOLO 類別:", data['yolo_classes'])
