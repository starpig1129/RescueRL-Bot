import zmq
import json
import struct
import time
import threading
import asyncio
import numpy as np
import cv2
import queue
from typing import Tuple, Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

class ServerManager:
    def __init__(self, logger=None):
        self.logger = logger
        self._init_zmq_vars()
        
        # 非同步相關變量
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=10)  # 增加線程池
        self.tasks = []  # 存放所有非同步任務
        self.running = True  # 控制執行狀態
        
        # 重置事件相關設置
        self.reset_event = threading.Event()
        self.reset_thread = None
        self.reset_thread_stop = threading.Event()
        
        # 當前世代
        self.epoch = 0
        
        # 數據緩存
        self.image_cache = None
        self.top_image_cache = None
        self.info_cache = None
        self.image_lock = threading.Lock()
        self.top_image_lock = threading.Lock()
        self.info_lock = threading.Lock()
        
        # 數據隊列
        self.info_queue = queue.Queue()  # 移除隊列大小限制，確保不丟失數據
        self.image_queue = queue.Queue()
        self.top_image_queue = queue.Queue()
        
        # 心跳檢測相關
        self.heartbeat_interval = 5.0  # 心跳檢測間隔（秒）
        self.heartbeat_timeout = 10.0  # 心跳超時時間（秒）
        self.connection_timeout = 30.0  # 連接超時時間（秒）
        self.last_heartbeat_time = {
            'control': 0, 'info': 0, 'obs': 0, 'reset': 0, 'top_camera': 0
        }
        self.heartbeat_thread = None

    def _init_zmq_vars(self):
        """初始化所有ZeroMQ相關變數"""
        # ZeroMQ 上下文
        self.context = None
        
        # ZeroMQ 套接字
        self.control_socket = None
        self.info_socket = None
        self.obs_socket = None
        self.reset_socket = None
        self.top_camera_socket = None
        
        # 定義所有伺服器地址
        self.control_address = 'tcp://*:5001'  # 修改端口
        self.info_address = 'tcp://*:8001'     # 修改端口
        self.obs_address = 'tcp://*:6001'      # 修改端口
        self.reset_address = 'tcp://*:7001'    # 修改端口
        self.top_camera_address = 'tcp://*:9001'  # 修改端口
        
        # 連接狀態
        self.control_connected = False
        # 初始設置為未連接
        self.info_connected = False
        # 初始設置為未連接
        self.obs_connected = False
        # 初始設置為未連接
        self.reset_connected = False
        self.top_camera_connected = False
        
        # 連接鎖
        self.control_lock = threading.Lock()
        self.info_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.reset_lock = threading.Lock()
        self.top_camera_lock = threading.Lock()

    def setup_all_servers(self):
        """設置並啟動所有伺服器"""
        try:
            # 創建 ZeroMQ 上下文
            self.context = zmq.Context()
            
            # 創建事件循環線程
            self.event_loop_thread = threading.Thread(target=self._run_event_loop)
            self.event_loop_thread.daemon = True
            self.event_loop_thread.start()
            
            # 設置所有伺服器
            asyncio.run_coroutine_threadsafe(self._setup_all_servers_async(), self.loop)
            
            # 等待所有伺服器套接字設置完成
            timeout = 30  # 30秒超時
            start_time = time.time()
            while not (self.control_socket is not None and 
                      self.info_socket is not None and 
                      self.obs_socket is not None and 
                      self.reset_socket is not None and 
                      self.top_camera_socket is not None):
                if time.time() - start_time > timeout:
                    raise TimeoutError("設置伺服器超時")
                
                # 等待套接字設置完成
                time.sleep(0.1)
                
            print("所有伺服器設置完成")
            
            # 添加延遲，解決ZeroMQ的"慢訂閱者問題"
            print("等待5秒，確保所有連接都有足夠的時間建立...")
            time.sleep(5)
            print("延遲結束")
            
            # 啟動數據接收線程
            self._start_data_receivers()
            
            # 啟動心跳檢測線程
            self._start_heartbeat_monitor()
            
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")
        
    def _run_event_loop(self):
        """在單獨的線程中運行事件循環"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            print(f"事件循環發生錯誤: {e}")
        finally:
            self.loop.close()

    async def _setup_all_servers_async(self):
        """非同步設置所有伺服器"""
        await asyncio.gather(
            self.setup_control_server(),
            self.setup_info_server(),
            self.setup_obs_server(),
            self.setup_top_camera_server(),
            self.setup_reset_server_async()
        )

    def _start_data_receivers(self):
        """啟動所有數據接收線程"""
        # 使用線程池接收數據
        self.info_receiver_thread = threading.Thread(target=self._info_receiver_loop)
        self.info_receiver_thread.daemon = True
        self.info_receiver_thread.start()
        
        self.obs_receiver_thread = threading.Thread(target=self._obs_receiver_loop)
        self.obs_receiver_thread.daemon = True
        self.obs_receiver_thread.start()
        
        self.top_camera_receiver_thread = threading.Thread(target=self._top_camera_receiver_loop)
        self.top_camera_receiver_thread.daemon = True
        self.top_camera_receiver_thread.start()
        
        # 啟動處理隊列的線程
        self.info_processor_thread = threading.Thread(target=self._info_processor_loop)
        self.info_processor_thread.daemon = True
        self.info_processor_thread.start()
        
        self.image_processor_thread = threading.Thread(target=self._image_processor_loop)
        self.image_processor_thread.daemon = True
        self.image_processor_thread.start()
        
    def _start_heartbeat_monitor(self):
        """啟動心跳監控線程"""
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        print("已啟動心跳監控")
    
    def _heartbeat_monitor_loop(self):
        """心跳監控線程函數"""
        while self.running:
            current_time = time.time()
            
            # 檢查各個連接的心跳狀態
            with self.control_lock:
                self.control_connected = (current_time - self.last_heartbeat_time.get('control', 0) < self.heartbeat_timeout)
            
            with self.info_lock:
                self.info_connected = (current_time - self.last_heartbeat_time.get('info', 0) < self.heartbeat_timeout)
            
            with self.obs_lock:
                self.obs_connected = (current_time - self.last_heartbeat_time.get('obs', 0) < self.heartbeat_timeout)
            
            with self.reset_lock:
                self.reset_connected = (current_time - self.last_heartbeat_time.get('reset', 0) < self.heartbeat_timeout)
            
            with self.top_camera_lock:
                self.top_camera_connected = (current_time - self.last_heartbeat_time.get('top_camera', 0) < self.heartbeat_timeout)
            
            time.sleep(1)  # 每秒檢查一次
    
    def _info_receiver_loop(self):
        """持續接收資訊數據的線程函數"""
        while self.running:
            try:
                with self.info_lock:
                    if not self.info_connected or self.info_socket is None:
                        time.sleep(0.1)
                        continue
                
                # 使用 ZeroMQ 接收數據
                try:
                    # 使用非阻塞模式接收數據
                    data = self._receive_info_internal()
                    if data:
                        # 更新心跳時間
                        if data.get('heartbeat', False):
                            self.last_heartbeat_time['info'] = time.time()
                            self.info_socket.send(b'ACK')  # 發送確認回應
                            continue
                        try:
                            # 將數據放入隊列，不再丟棄數據
                            if self.info_queue.full():
                                try:
                                    self.info_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.info_queue.put_nowait(data)
                        except Exception as e:
                            # 即使處理失敗也要發送回應
                            self.info_socket.send(b'ACK')
                            print(f"將資訊數據放入隊列時發生錯誤: {e}")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # 沒有數據可接收，繼續等待
                        time.sleep(0.01)
                    else:
                        print(f"ZeroMQ 接收資訊數據時發生錯誤: {e}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"接收資訊數據時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _obs_receiver_loop(self):
        """持續接收觀察圖像的線程函數"""
        while self.running:
            try:
                with self.obs_lock:
                    if not self.obs_connected or self.obs_socket is None:
                        time.sleep(0.1)
                        continue
                
                # 使用 ZeroMQ 接收數據
                try:
                    # 使用非阻塞模式接收數據
                    image = self._receive_image_internal(self.obs_socket)
                    if image is not None:
                        # 更新心跳時間
                        self.last_heartbeat_time['obs'] = time.time()
                        self.obs_socket.send(b'ACK')  # 發送確認回應
                        
                        # 處理圖像數據
                        try:
                            # 將圖像放入隊列，不再丟棄數據
                            if self.image_queue.full():
                                try:
                                    self.image_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.image_queue.put_nowait(image)
                        except Exception as e:
                            # 即使處理失敗也要發送回應
                            self.obs_socket.send(b'ACK')
                            print(f"將觀察圖像放入隊列時發生錯誤: {e}")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # 沒有數據可接收，繼續等待
                        time.sleep(0.01)
                    else:
                        print(f"ZeroMQ 接收觀察圖像時發生錯誤: {e}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"接收觀察圖像時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _top_camera_receiver_loop(self):
        """持續接收頂部攝影機圖像的線程函數"""
        while self.running:
            try:
                with self.top_camera_lock:
                    if not self.top_camera_connected or self.top_camera_socket is None:
                        time.sleep(0.1)
                        continue
                
                # 使用 ZeroMQ 接收數據
                try:
                    # 使用非阻塞模式接收數據
                    image = self._receive_image_internal(self.top_camera_socket)
                    if image is not None:
                        # 更新心跳時間
                        self.last_heartbeat_time['top_camera'] = time.time()
                        self.top_camera_socket.send(b'ACK')  # 發送確認回應
                        
                        # 處理圖像數據
                        try:
                            # 將圖像放入隊列，不再丟棄數據
                            if self.top_image_queue.full():
                                try:
                                    self.top_image_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.top_image_queue.put_nowait(image)
                        except Exception as e:
                            # 即使處理失敗也要發送回應
                            self.top_camera_socket.send(b'ACK')
                            print(f"將頂部攝影機圖像放入隊列時發生錯誤: {e}")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # 沒有數據可接收，繼續等待
                        time.sleep(0.01)
                    else:
                        print(f"ZeroMQ 接收頂部攝影機圖像時發生錯誤: {e}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"接收頂部攝影機圖像時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _info_processor_loop(self):
        """處理資訊數據隊列的線程函數"""
        while self.running:
            try:
                # 從隊列中獲取數據
                data = self.info_queue.get(timeout=0.5)
                # 更新緩存
                with self.info_lock:
                    self.info_cache = data
                
                # 標記任務完成
                self.info_queue.task_done()
            except queue.Empty:
                # 隊列為空，繼續等待
                continue
            except Exception as e:
                print(f"處理資訊數據時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _image_processor_loop(self):
        """處理圖像數據隊列的線程函數"""
        while self.running:
            try:
                # 嘗試從觀察圖像隊列獲取數據
                try:
                    image = self.image_queue.get(block=False)
                    with self.image_lock:
                        self.image_cache = image
                    self.image_queue.task_done()
                except queue.Empty:
                    pass
                
                # 嘗試從頂部攝影機圖像隊列獲取數據
                try:
                    image = self.top_image_queue.get(block=False)
                    with self.top_image_lock:
                        self.top_image_cache = image
                    self.top_image_queue.task_done()
                except queue.Empty:
                    pass
                
                # 短暫休眠，避免CPU使用率過高
                time.sleep(0.01)
            except Exception as e:
                print(f"處理圖像數據時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _receive_info_internal(self) -> Optional[Dict[str, Any]]:
        """內部方法：接收資訊數據"""
        try:
            # 使用 ZeroMQ 接收數據
            try:
                # 使用非阻塞模式接收數據
                message = self.info_socket.recv(flags=zmq.NOBLOCK)
                # REP模式
                
                # 解析 JSON 數據
                data_str = message.decode('utf-8')
                return json.loads(data_str)
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    # 沒有數據可接收
                    return None
                else:
                    raise
        except Exception as e:
            print(f"接收資訊數據時發生錯誤: {e}")
            return None
    
    def _receive_image_internal(self, socket: zmq.Socket) -> Optional[np.ndarray]:
        """內部方法：接收圖像數據"""
        try:
            # 使用 ZeroMQ 接收數據
            try:
                # 使用非阻塞模式接收數據
                message = socket.recv(flags=zmq.NOBLOCK)
                # REP模式

                # 解析圖像數據
                nparr = np.frombuffer(message, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    # 沒有數據可接收
                    return None
                else:
                    raise
        except Exception as e:
            print(f"接收圖像數據時發生錯誤: {e}")
            return None
    
    async def setup_reset_server_async(self):
        """非同步設置重置伺服器"""
        try:
            # 創建 REP 套接字 (回應模式)
            self.reset_socket = self.context.socket(zmq.REP)
            self.reset_socket.bind(self.reset_address)
            print("重置訊號伺服器已啟動，等待連接...")
            
            # 設置非阻塞模式
            self.reset_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超時
            
            # 啟動重置連接處理線程
            self.reset_thread = threading.Thread(target=self._handle_reset_requests)
            self.reset_thread.daemon = True
            self.reset_thread.start()
            
            # 標記為已連接
            self.reset_connected = True  # 初始設置為已連接，給予一個寬限期
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")
    
    async def setup_top_camera_server(self):
        """設置頂部攝影機伺服器"""
        try:
            # 創建 SUB 套接字 (訂閱模式)
            self.top_camera_socket = self.context.socket(zmq.REP)
            self.top_camera_socket.bind(self.top_camera_address)
            self.top_camera_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有消息
            print("頂部攝影機伺服器已啟動，等待連接...")
            
            # 設置非阻塞模式
            self.top_camera_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100毫秒超時
            
            # 標記為已連接
            self.top_camera_connected = True  # 初始設置為已連接，給予一個寬限期
        except Exception as e:
            raise Exception(f"頂部攝影機伺服器設置失敗: {e}")

    async def setup_control_server(self):
        """設置控制伺服器"""
        try:
            # 創建 PULL 套接字 (接收模式)
            self.control_socket = self.context.socket(zmq.REP)
            self.control_socket.bind(self.control_address)
            print('控制伺服器已啟動，等待連接...')
            
            # 設置非阻塞模式
            self.control_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100毫秒超時
            
            # 標記為已連接
            self.control_connected = True  # 初始設置為已連接，給予一個寬限期
        except Exception as e:
            raise Exception(f"控制伺服器設置失敗: {e}")

    async def setup_info_server(self):
        """設置資訊伺服器"""
        try:
            # 創建 SUB 套接字 (訂閱模式)
            self.info_socket = self.context.socket(zmq.REP)
            self.info_socket.bind(self.info_address)
            self.info_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有消息
            print('資訊伺服器已啟動，等待連接...')
            
            # 設置非阻塞模式
            self.info_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100毫秒超時
            
            # 標記為已連接
            self.info_connected = True  # 初始設置為已連接，給予一個寬限期
        except Exception as e:
            raise Exception(f"資訊伺服器設置失敗: {e}")

    async def setup_obs_server(self):
        """設置觀察伺服器"""
        try:
            # 創建 SUB 套接字 (訂閱模式)
            self.obs_socket = self.context.socket(zmq.REP)
            self.obs_socket.bind(self.obs_address)
            self.obs_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 訂閱所有消息
            print("影像接收伺服器已啟動，等待連接...")
            
            # 設置非阻塞模式
            self.obs_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100毫秒超時
            
            # 標記為已連接
            self.obs_connected = True  # 初始設置為已連接，給予一個寬限期
        except Exception as e:
            raise Exception(f"影像接收伺服器設置失敗: {e}")

    
    def handle_reset_requests(self):
        """處理重置請求的線程函數"""
        while not self.reset_thread_stop.is_set():
            try:
                # 嘗試接收重置信號
                try:
                    # 使用非阻塞模式接收數據
                    print("等待接收重置信號...")
                    message = self.reset_socket.recv(flags=zmq.NOBLOCK)
                    
                    # 解析重置信號
                    print(f"收到重置信號: {len(message)} 字節")
                    signal = int.from_bytes(message, byteorder='little')
                    
                    # 更新心跳時間
                    self.last_heartbeat_time['reset'] = time.time()
                    
                    if signal == 1:
                        # 增加 epoch 值
                        print(f"處理重置信號,當前epoch: {self.epoch}")
                        self.epoch += 1
                        epoch_data = self.epoch.to_bytes(4, byteorder='little')
                        
                        # 發送 epoch 值
                        self.reset_socket.send(epoch_data)
                        
                        print(f"已發送 epoch 回應: {self.epoch}")
                        # 設置重置事件
                        self.reset_event.set()
                        print(f"收到重置訊號，當前 epoch: {self.epoch}")
                    else:
                        # 對於其他信號,也需要發送回應
                        print(f"收到未知信號: {signal}, 發送默認回應")
                        try:
                            self.reset_socket.send(struct.pack('i', -1))
                        except Exception as e:
                            print(f"發送回應失敗: {e}")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # 沒有數據可接收，繼續等待
                        time.sleep(0.01)
                    else:
                        # 即使發生錯誤也要嘗試發送回應
                        try:
                            # 檢查是否已經發送過回應
                            if e.errno != zmq.EFSM:  # EFSM表示狀態機錯誤,可能已經發送過回應
                                print("嘗試發送錯誤回應...")
                                self.reset_socket.send(struct.pack('i', -1))  # 發送錯誤回應
                                print("錯誤回應已發送")
                        except: pass
                        print(f"ZeroMQ 接收重置信號時發生錯誤: {e}")
                        time.sleep(0.1)
            except Exception as e:
                print(f"處理重置請求時發生錯誤: {e}")
                time.sleep(0.1)  
    # 為了向後兼容，保留舊方法名稱
    def _handle_reset_requests(self):
        return self.handle_reset_requests()

    def send_control_signal(self, relative_angle: float) -> None:
        """發送控制信號到Unity"""
        try:
            with self.control_lock:
                # 檢查連接狀態
                if not self.control_connected or self.control_socket is None:
                    print("控制連接未建立，無法發送控制信號")
                    return
                
                # 在REP模式下，需要先接收請求，然後發送回應
                try:
                    # 嘗試接收請求
                    request = self.control_socket.recv(flags=zmq.NOBLOCK)
  # 非阻塞接收
                    print(f"收到控制請求: {len(request)} 字節")
                    
                    # 使用 ZeroMQ 發送數據
                    buffer = struct.pack('f', relative_angle)
                
    
                    # 更新心跳時間
                    self.last_heartbeat_time['control'] = time.time()
                    # 記錄最後一次發送控制信號的時間
                    self.control_socket.send(buffer)
                    print(f"已發送控制回應: {relative_angle}")
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        # 沒有請求可接收，這是正常的
                        time.sleep(0.01)  # 短暫等待
                    else:
                        print(f"控制信號處理錯誤: {e}")
        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")

    def receive_image(self, show: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """接收影像數據 (使用緩存)"""
        with self.image_lock:
            image = self.image_cache
            
            # 如果沒有收到過圖像數據，返回 None
            if image is None:
                print("警告：尚未收到任何圖像數據，Unity 客戶端可能未連接")
        
        if image is None:
            return None, None
        
        if show:
            cv2.imshow("觀察空間", image)
            cv2.waitKey(1)
        
        return image, image.copy() if image is not None else None

    def receive_top_camera_image(self, show: bool = False) -> Optional[np.ndarray]:
        """接收頂部攝影機的影像 (使用緩存)"""
        with self.top_image_lock:
            image = self.top_image_cache
            
            # 如果沒有收到過圖像數據，返回 None
            if image is None:
                print("警告：尚未收到任何頂部攝影機圖像數據，Unity 客戶端可能未連接")
        
        if image is None:
            return None
        
        if show:
            cv2.imshow("頂部視角", image)
            cv2.waitKey(1)
        
        return image

    def receive_info(self) -> Optional[Dict[str, Any]]:
        """接收資訊數據 (使用緩存)"""
        with self.info_lock:
            info = self.info_cache
            
            # 如果沒有收到過資訊數據，返回 None
            if info is None:
                print("警告：尚未收到任何資訊數據，Unity 客戶端可能未連接")
        
        return info

    def set_epoch(self, epoch: int) -> None:
        """設置當前世代"""
        self.epoch = epoch

    def is_reset_triggered(self) -> bool:
        """檢查是否觸發重置"""
        return self.reset_event.is_set()

    def clear_reset_event(self) -> None:
        """清除重置事件"""
        self.reset_event.clear()

    def is_unity_connected(self) -> bool:
        """檢查 Unity 客戶端是否已連接"""
        # 檢查是否收到過任何數據
        has_received_data = (
            self.image_cache is not None or
            self.top_image_cache is not None or
            self.info_cache is not None
        )
        
        # 檢查心跳狀態
        return has_received_data and (self.control_connected or self.info_connected or self.obs_connected)

    def close(self) -> None:
        """關閉所有連接和資源"""
        print("正在關閉伺服器管理器...")
        
        try:
            # 關閉非同步任務
            if hasattr(self, 'tasks') and self.tasks:
                for task in self.tasks:
                    if not task.done():
                        task.cancel()
                
                if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
                    try:
                        pending = asyncio.all_tasks(self.loop)
                        if pending:
                            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception as e:
                        print(f"關閉非同步任務時發生錯誤: {e}")
        except Exception as e:
            print(f"清理非同步任務時發生錯誤: {e}")
            
        try:
            # 關閉重置線程
            if hasattr(self, 'reset_thread_stop'):
                self.reset_thread_stop.set()
                
            # 關閉心跳監控線程
            if hasattr(self, 'heartbeat_thread') and self.heartbeat_thread is not None:
                self.reset_thread_stop.set()
            
            if hasattr(self, 'reset_thread') and self.reset_thread is not None:
                self.reset_thread.join(timeout=1)
        except Exception as e:
            print(f"關閉重置線程時發生錯誤: {e}")
            
        try:
            # 關閉所有 ZeroMQ 套接字
            sockets = [
                'control_socket',
                'info_socket',
                'obs_socket',
                'reset_socket',
                'top_camera_socket'
            ]
            
            for socket_attr in sockets:
                if hasattr(self, socket_attr) and getattr(self, socket_attr) is not None:
                    try:
                        socket = getattr(self, socket_attr)
                        socket.close(linger=0)
                    except Exception as e:
                        print(f"關閉 {socket_attr} 時發生錯誤: {e}")
                    finally:
                        setattr(self, socket_attr, None)
        except Exception as e:
            print(f"關閉 ZeroMQ 套接字時發生錯誤: {e}")
            
        try:
            # 關閉 ZeroMQ 上下文
            if hasattr(self, 'context') and self.context is not None:
                try:
                    self.context.term()
                except Exception as e:
                    print(f"關閉 ZeroMQ 上下文時發生錯誤: {e}")
                finally:
                    self.context = None
        except Exception as e:
            print(f"清理 ZeroMQ 上下文時發生錯誤: {e}")
            
        try:
            # 關閉事件循環
            if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
                try:
                    self.loop.stop()
                    self.loop.close()
                except Exception as e:
                    print(f"關閉事件循環時發生錯誤: {e}")
                finally:
                    self.loop = None
        except Exception as e:
            print(f"清理事件循環時發生錯誤: {e}")
        
        print("伺服器管理器關閉完成")