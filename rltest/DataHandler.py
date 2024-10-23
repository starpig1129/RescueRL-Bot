import os
import h5py
import numpy as np
import threading
import queue

class DataHandler:
    def __init__(self, base_dir="data", resize_step=1):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.hdf5_file = None
        self.current_max_steps = 0  # 當前最大步數（從 0 開始）
        self.resize_step = resize_step  # 每次需要擴展時增加的步數
        self.write_queue = queue.Queue()  # 建立一個 Queue 用於異步寫入
        self.stop_event = threading.Event()  # 用於通知異步寫入執行緒結束
        self.writer_thread = None  # 異步寫入的執行緒

    def create_epoch_file(self, epoch):
        """
        創建一個 HDF5 檔案來儲存整個世代的資料，無預設步數大小
        :param epoch: 世代號碼
        """
        # 關閉當前的檔案和執行緒
        self.close_epoch_file()
        # 重置步數
        self.current_max_steps = 0
        file_path = os.path.join(self.base_dir, f"epoch_{epoch:03d}.h5")
        self.hdf5_file = h5py.File(file_path, 'w')

        # 使用 chunking 並啟用可擴展的資料集
        chunk_size = (100, 384, 640, 3)  # 設定合理的 chunk 大小

        # 創建資料集，初始大小為 (0, ...) 並啟用 maxshape，允許動態擴展
        self.obs_dataset = self.hdf5_file.create_dataset(
            'obs', (0, 384, 640, 3), maxshape=(None, 384, 640, 3), dtype=np.uint8, chunks=chunk_size)
        self.angle_dataset = self.hdf5_file.create_dataset(
            'angle_degrees', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
        self.reward_dataset = self.hdf5_file.create_dataset(
            'reward', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
        
        # reward_list 改為 12 長度，並且是可擴展的
        self.reward_list_dataset = self.hdf5_file.create_dataset(
            'reward_list', (0, 12), maxshape=(None, 12), dtype=np.float32, chunks=True)
        self.origin_image_dataset = self.hdf5_file.create_dataset(
            'origin_image', (0, 384, 640, 3), maxshape=(None, 384, 640, 3), dtype=np.uint8, chunks=chunk_size)
        self.yolo_boxes_dataset = self.hdf5_file.create_dataset(
            'yolo_boxes', (0, 10, 4), maxshape=(None, 100, 4), dtype=np.float32, chunks=True)
        self.yolo_scores_dataset = self.hdf5_file.create_dataset(
            'yolo_scores', (0, 10), maxshape=(None, 100), dtype=np.float32, chunks=True)
        self.yolo_classes_dataset = self.hdf5_file.create_dataset(
            'yolo_classes', (0, 10), maxshape=(None, 100), dtype=np.int32, chunks=True)
        
        print(f"資料檔案已創建並啟用可擴展: {file_path}")

        # 啟動異步寫入執行緒
        self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
        self.writer_thread.start()

    def save_step_data(self, step, obs, angle_degrees, reward, reward_list, origin_image, results):
        """
        儲存每個步數的資料到 HDF5 檔案的 Queue 中，並由異步執行緒進行寫入
        """
        # 如果步數超過了當前的最大步數，則動態擴展資料集
        if step >= self.current_max_steps:
            self._resize_datasets()

        # 將步數的所有資料加入到寫入隊列中
        self.write_queue.put((step, obs, angle_degrees, reward, reward_list, origin_image, results))

    def _resize_datasets(self):
        """
        動態擴展所有資料集的大小
        """
        new_max_steps = self.current_max_steps + self.resize_step
        print(f"擴展資料集大小到 {new_max_steps} 步數")

        # 使用 maxshape 屬性來調整資料集大小
        self.obs_dataset.resize(new_max_steps, axis=0)
        self.angle_dataset.resize(new_max_steps, axis=0)
        self.reward_dataset.resize(new_max_steps, axis=0)
        self.reward_list_dataset.resize(new_max_steps, axis=0)
        self.origin_image_dataset.resize(new_max_steps, axis=0)
        self.yolo_boxes_dataset.resize(new_max_steps, axis=0)
        self.yolo_scores_dataset.resize(new_max_steps, axis=0)
        self.yolo_classes_dataset.resize(new_max_steps, axis=0)

        # 更新當前最大步數
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

    def _write_data_to_hdf5(self, step, obs, angle_degrees, reward, reward_list, origin_image, results):
        """
        將數據寫入 HDF5 檔案
        """
        # 寫入各種數據到對應的資料集
        self.obs_dataset[step] = obs
        self.angle_dataset[step] = angle_degrees
        self.reward_dataset[step] = reward
        self.reward_list_dataset[step] = reward_list
        self.origin_image_dataset[step] = origin_image

        if results is not None and len(results) > 0 and results[0].boxes:
            yolo_boxes = np.array(results[0].boxes.xyxy)
            yolo_scores = np.array(results[0].boxes.conf)
            yolo_classes = np.array(results[0].boxes.cls)
            self.yolo_boxes_dataset[step, :len(yolo_boxes)] = yolo_boxes
            self.yolo_scores_dataset[step, :len(yolo_scores)] = yolo_scores
            self.yolo_classes_dataset[step, :len(yolo_classes)] = yolo_classes
        else:
            self.yolo_boxes_dataset[step] = np.zeros((10, 4))
            self.yolo_scores_dataset[step] = np.zeros(10)
            self.yolo_classes_dataset[step] = np.zeros(10, dtype=np.int32)

        print(f"步數 {step} 的資料已儲存到 HDF5 檔案")

    def close_epoch_file(self):
        # 發送停止訊號並等待寫入執行緒結束
        self.stop_event.set()
        if self.writer_thread is not None:
            self.writer_thread.join()

        # 清空寫入隊列，避免未寫入完畢的數據影響下一個世代
        with self.write_queue.mutex:
            self.write_queue.queue.clear()

        # 確保寫入資料完成後關閉 HDF5 檔案
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            print("HDF5 檔案已關閉")

        # 重置停止事件和寫入執行緒狀態
        self.stop_event.clear()
        self.writer_thread = None

