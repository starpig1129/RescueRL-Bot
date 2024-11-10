import os
import h5py
import numpy as np

class DataReader:
    def __init__(self, base_dir="train_logs"):
        self.base_dir = base_dir
        self.env_dir = os.path.join(base_dir, "env_data")
        self.feature_dir = os.path.join(base_dir, "feature_data")
        self.feature_save_interval = 10  # 特徵保存間隔
        
    def get_max_steps(self, epoch):
        """讀取指定世代的最大步數"""
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        
        if not os.path.exists(env_file_path):
            print(f"檔案 {env_file_path} 不存在")
            return None
        
        try:
            with h5py.File(env_file_path, 'r') as env_file:
                if 'obs' not in env_file:
                    print(f"資料集 'obs' 在 {env_file_path} 中不存在")
                    return None
                
                max_steps = env_file['obs'].shape[0]
                return max_steps
                
        except Exception as e:
            print(f"讀取檔案時發生錯誤: {e}")
            return None

    def _get_feature_data_for_step(self, feature_file, feature_step, feature_datasets):
        """獲取特定step的特徵數據，如果不存在則返回None"""
        try:
            # 檢查step_mapping中是否有對應的數據
            if 'step_mapping' in feature_file:
                mappings = feature_file['step_mapping'][:]
                if feature_step < len(mappings):
                    feature_data = {}
                    for name in feature_datasets:
                        if name != 'step_mapping':
                            feature_data[name] = feature_file[name][feature_step]
                    return feature_data
            return None
        except Exception as e:
            print(f"讀取特徵數據時發生錯誤: {e}")
            return None

    def load_range_data(self, epoch, slice_obj):
        """使用 NumPy 樣式的切片方式讀取指定世代和步數範圍的資料，自動對齊環境和特徵數據"""
        env_file_path = os.path.join(self.env_dir, f"ep{epoch:03d}_env.h5")
        feature_file_path = os.path.join(self.feature_dir, f"ep{epoch:03d}_feature.h5")
        
        if not os.path.exists(env_file_path):
            print(f"環境數據檔案 {env_file_path} 不存在")
            return None
            
        if not os.path.exists(feature_file_path):
            print(f"特徵數據檔案 {feature_file_path} 不存在")
            return None
        
        try:
            # 準備返回的數據字典
            aligned_data = {}
            
            # 打開環境數據檔案
            with h5py.File(env_file_path, 'r') as env_file:
                # 檢查並讀取環境數據集
                env_datasets = [
                    'obs', 'angle_degrees', 'reward', 'reward_list',
                    'origin_image', 'yolo_boxes', 'yolo_scores', 'yolo_classes'
                ]
                
                for dataset in env_datasets:
                    if dataset not in env_file:
                        print(f"環境資料集 {dataset} 在 {env_file_path} 中不存在")
                        return None
                    aligned_data[dataset] = env_file[dataset][slice_obj]
                
                # 獲取實際的步數範圍
                if isinstance(slice_obj, slice):
                    start = slice_obj.start if slice_obj.start is not None else 0
                    stop = slice_obj.stop if slice_obj.stop is not None else len(env_file['obs'])
                    steps = range(start, stop)
                else:
                    steps = [slice_obj]
            
            # 打開特徵數據檔案
            with h5py.File(feature_file_path, 'r') as feature_file:
                # 檢查特徵數據集
                feature_datasets = [
                    'layer_input', 'layer_conv1', 'layer_final_residual',
                    'layer_feature', 'layer_actor'
                ]
                
                # 為每個step創建對應的特徵數據字典
                for dataset in feature_datasets:
                    if dataset not in feature_file:
                        print(f"特徵資料集 {dataset} 在 {feature_file_path} 中不存在")
                        return None
                    # 初始化為None陣列
                    aligned_data[dataset] = np.array([None] * len(steps), dtype=object)
                
                # 讀取step_mapping
                if 'step_mapping' in feature_file:
                    step_mappings = feature_file['step_mapping'][:]
                    
                    # 填充特徵數據
                    for i, step in enumerate(steps):
                        feature_step = step // self.feature_save_interval
                        if feature_step < len(step_mappings) and step_mappings[feature_step] == step:
                            # 該步數有特徵數據，讀取所有特徵
                            for dataset in feature_datasets:
                                aligned_data[dataset][i] = feature_file[dataset][feature_step]
            
            print(f"成功讀取世代 {epoch} 的資料範圍，環境和特徵數據已對齊")
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
                'total_steps': 0,
                'total_feature_steps': 0
            }
            
            # 獲取環境數據集的信息
            with h5py.File(env_file_path, 'r') as env_file:
                for key in env_file.keys():
                    dataset = env_file[key]
                    info['env_datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size': dataset.size
                    }
                    if key == 'obs':
                        info['total_steps'] = dataset.shape[0]
            
            # 獲取特徵數據集的信息
            with h5py.File(feature_file_path, 'r') as feature_file:
                for key in feature_file.keys():
                    dataset = feature_file[key]
                    info['feature_datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size': dataset.size
                    }
                    if key == 'step_mapping':
                        info['total_feature_steps'] = dataset.shape[0]
            
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
            print(f"總環境步數: {file_info['total_steps']}")
            print(f"總特徵步數: {file_info['total_feature_steps']}")
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
            data_range = data_reader.load_range_data(test_epoch, slice(0, 3000))
            if data_range:
                print("\n成功讀取數據範圍:")
                for key, value in data_range.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype == object:
                            # 計算非None值的數量
                            non_none_count = np.sum([x is not None for x in value])
                            print(f"{key}: shape={value.shape}, 非None值數量={non_none_count}")
                        else:
                            print(f"{key}: shape={value.shape}, dtype={value.dtype}")