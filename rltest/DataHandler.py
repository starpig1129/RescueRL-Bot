import os
import h5py
import numpy as np

class DataHandler:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.hdf5_file = None

    def create_epoch_file(self, epoch):
        """
        創建一個 HDF5 檔案來儲存整個世代的資料
        :param epoch: 世代號碼
        """
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        self.hdf5_file = h5py.File(file_path, 'w')
        print(f"資料檔案已創建: {file_path}")

    def save_step_data(self, step, obs, angle_degrees, reward, reward_list, origin_image, results):
        """
        儲存每個步數的資料到 HDF5 檔案中
        :param step: 當前步數
        :param obs: 觀察空間
        :param angle_degrees: 動作對應的角度
        :param reward: 獎勵值
        :param reward_list: 獎勵列表
        :param origin_image: 原始攝影機畫面
        :param results: YOLOv8 模型回傳結果
        """
        if self.hdf5_file is not None:
            step_group = self.hdf5_file.create_group(f"step_{step:03d}")
            
            # 儲存步數的資料
            step_group.create_dataset('obs', data=obs)  # 儲存觀察空間
            step_group.create_dataset('angle_degrees', data=angle_degrees)  # 儲存動作角度
            step_group.create_dataset('reward', data=reward)  # 儲存獎勵值
            step_group.create_dataset('reward_list', data=np.array(reward_list))  # 儲存獎勵清單
            step_group.create_dataset('origin_image', data=np.array(origin_image))  # 儲存原始影像

            # 檢查 YOLOv8 的結果，避免空的結果導致錯誤
            if results is not None and len(results) > 0 and results[0].boxes:
                yolo_boxes = np.array(results[0].boxes.xyxy)
                yolo_scores = np.array(results[0].boxes.conf)
                yolo_classes = np.array(results[0].boxes.cls)
            else:
                # 若沒有偵測到物件，填入空值
                yolo_boxes = np.array([])  
                yolo_scores = np.array([])  
                yolo_classes = np.array([])  

            # 儲存 YOLO 偵測結果
            step_group.create_dataset('yolo_boxes', data=yolo_boxes)
            step_group.create_dataset('yolo_scores', data=yolo_scores)
            step_group.create_dataset('yolo_classes', data=yolo_classes)

            print(f"步數 {step} 的資料已儲存")
        else:
            print("HDF5 檔案尚未創建，無法儲存資料")




    def close_epoch_file(self):
        """
        關閉 HDF5 檔案
        """
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            print("HDF5 檔案已關閉")
            self.hdf5_file = None
