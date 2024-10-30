import os
import h5py
import numpy as np

class DataReader:
    def __init__(self, base_dir="data"):
        """
        初始化 DataReader
        
        Parameters:
            base_dir (str): 數據文件的基礎目錄
        """
        self.base_dir = base_dir

    def get_max_steps(self, epoch):
        """
        讀取指定世代的最大步數
        
        Parameters:
            epoch (int): 世代號碼
            
        Returns:
            int: 最大步數，如果讀取失敗則返回 None
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        try:
            with h5py.File(file_path, 'r') as hdf5_file:
                if 'obs' not in hdf5_file:
                    print(f"資料集 'obs' 在 {file_path} 中不存在")
                    return None
                
                max_steps = hdf5_file['obs'].shape[0]
                return max_steps
                
        except Exception as e:
            print(f"讀取檔案時發生錯誤: {e}")
            return None

    def load_range_data(self, epoch, slice_obj):
        """
        使用 NumPy 樣式的切片方式讀取指定世代和步數範圍的資料
        
        Parameters:
            epoch (int): 世代號碼
            slice_obj: NumPy 樣式的切片物件
            
        Returns:
            dict: 包含所有數據的字典，如果讀取失敗則返回 None
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        try:
            with h5py.File(file_path, 'r') as hdf5_file:
                # 檢查基本數據集是否存在
                basic_datasets = [
                    'obs', 'angle_degrees', 'reward', 'reward_list',
                    'origin_image', 'yolo_boxes', 'yolo_scores', 'yolo_classes'
                ]
                
                # 檢查神經網路層輸出數據集是否存在
                layer_datasets = [
                    'layer_input', 'layer_conv1', 'layer_final_residual',
                    'layer_feature', 'layer_actor'
                ]
                
                # 檢查所有必需的數據集
                for dataset in basic_datasets + layer_datasets:
                    if dataset not in hdf5_file:
                        print(f"資料集 {dataset} 在 {file_path} 中不存在")
                        return None
                
                # 讀取基本數據
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
                
                # 讀取神經網路層輸出
                layer_data = {
                    'layer_input': hdf5_file['layer_input'][slice_obj],
                    'layer_conv1': hdf5_file['layer_conv1'][slice_obj],
                    'layer_final_residual': hdf5_file['layer_final_residual'][slice_obj],
                    'layer_feature': hdf5_file['layer_feature'][slice_obj],
                    'layer_actor': hdf5_file['layer_actor'][slice_obj]
                }
                
                # 合併基本數據和層輸出數據
                data.update(layer_data)
                
                print(f"成功讀取世代 {epoch} 的資料範圍")
                return data
                
        except Exception as e:
            print(f"讀取資料時發生錯誤: {e}")
            return None
    
    def get_all_epochs(self):
        """
        獲取目錄中所有可用的世代號碼
        
        Returns:
            list: 排序後的世代號碼列表
        """
        try:
            h5_files = [f for f in os.listdir(self.base_dir) if f.endswith('.h5')]
            epochs = []
            for f in h5_files:
                try:
                    epoch = int(f.split('_')[1].split('.')[0])
                    epochs.append(epoch)
                except:
                    continue
            return sorted(epochs)
        except Exception as e:
            print(f"獲取世代列表時發生錯誤: {e}")
            return []
    
    def get_file_info(self, epoch):
        """
        獲取指定世代數據文件的詳細信息
        
        Parameters:
            epoch (int): 世代號碼
            
        Returns:
            dict: 包含文件信息的字典
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        
        if not os.path.exists(file_path):
            print(f"檔案 {file_path} 不存在")
            return None
        
        try:
            with h5py.File(file_path, 'r') as hdf5_file:
                info = {
                    'file_size': os.path.getsize(file_path) / (1024 * 1024),  # MB
                    'datasets': {},
                    'total_steps': 0
                }
                
                # 獲取所有數據集的信息
                for key in hdf5_file.keys():
                    dataset = hdf5_file[key]
                    info['datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size': dataset.size
                    }
                    if key == 'obs':
                        info['total_steps'] = dataset.shape[0]
                
                return info
                
        except Exception as e:
            print(f"獲取檔案信息時發生錯誤: {e}")
            return None

if __name__ == "__main__":
    # 使用範例
    data_reader = DataReader(base_dir="test_logs")
    
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
            print(f"檔案大小: {file_info['file_size']:.2f} MB")
            print(f"總步數: {file_info['total_steps']}")
            print("\n數據集信息:")
            for name, info in file_info['datasets'].items():
                print(f"{name}: shape={info['shape']}, dtype={info['dtype']}")
        
        # 讀取部分數據
        max_steps = data_reader.get_max_steps(test_epoch)
        if max_steps:
            print(f"\n世代 {test_epoch} 的最大步數: {max_steps}")
            
            # 讀取前 10 步的數據
            data_range = data_reader.load_range_data(test_epoch, slice(0, 10))
            if data_range:
                print("\n成功讀取數據範圍:")
                for key, value in data_range.items():
                    print(f"{key} shape: {value.shape}")
                print(data_range['layer_actor'][0])