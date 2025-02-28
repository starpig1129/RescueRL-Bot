import socket
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
        self._init_socket_vars()
        
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
        self.info_queue = queue.Queue(maxsize=10)  # 限制隊列大小，避免內存溢出
        self.image_queue = queue.Queue(maxsize=5)
        self.top_image_queue = queue.Queue(maxsize=5)

    def _init_socket_vars(self):
        """初始化所有socket相關變數"""
        self.control_socket = None
        self.control_conn = None
        self.info_socket = None
        self.info_conn = None
        self.obs_socket = None
        self.obs_conn = None
        self.reset_socket = None
        self.reset_conn = None
        self.top_camera_socket = None
        self.top_camera_conn = None
        
        # 定義所有伺服器地址
        self.control_address = ('localhost', 5000)
        self.info_address = ('localhost', 8000)
        self.obs_address = ('localhost', 6000)
        self.reset_address = ('localhost', 7000)
        self.top_camera_address = ('localhost', 9000)
        
        # 連接狀態
        self.control_connected = False
        self.info_connected = False
        self.obs_connected = False
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
            # 創建事件循環線程
            self.event_loop_thread = threading.Thread(target=self._run_event_loop)
            self.event_loop_thread.daemon = True
            self.event_loop_thread.start()
            
            # 設置所有伺服器
            asyncio.run_coroutine_threadsafe(self._setup_all_servers_async(), self.loop)
            
            # 等待所有伺服器設置完成
            timeout = 30  # 30秒超時
            start_time = time.time()
            while not (self.control_connected and self.info_connected and 
                      self.obs_connected and self.reset_connected and 
                      self.top_camera_connected):
                if time.time() - start_time > timeout:
                    raise TimeoutError("設置伺服器超時")
                time.sleep(0.1)
                
            print("所有伺服器設置完成")
            
            # 啟動數據接收線程
            self._start_data_receivers()
            
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
                    
    def _info_receiver_loop(self):
        """持續接收資訊數據的線程函數"""
        while self.running:
            try:
                with self.info_lock:
                    if not self.info_connected or self.info_conn is None:
                        time.sleep(0.1)
                        continue
                
                data = self._receive_info_internal()
                if data:
                    try:
                        # 將數據放入隊列，如果隊列已滿則丟棄最舊的數據
                        if self.info_queue.full():
                            try:
                                self.info_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.info_queue.put_nowait(data)
                    except Exception as e:
                        print(f"將資訊數據放入隊列時發生錯誤: {e}")
            except Exception as e:
                print(f"接收資訊數據時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _obs_receiver_loop(self):
        """持續接收觀察圖像的線程函數"""
        while self.running:
            try:
                with self.obs_lock:
                    if not self.obs_connected or self.obs_conn is None:
                        time.sleep(0.1)
                        continue
                
                image = self._receive_image_internal(self.obs_conn)
                if image is not None:
                    try:
                        # 將圖像放入隊列，如果隊列已滿則丟棄最舊的圖像
                        if self.image_queue.full():
                            try:
                                self.image_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.image_queue.put_nowait(image)
                    except Exception as e:
                        print(f"將觀察圖像放入隊列時發生錯誤: {e}")
            except Exception as e:
                print(f"接收觀察圖像時發生錯誤: {e}")
                time.sleep(0.1)
    
    def _top_camera_receiver_loop(self):
        """持續接收頂部攝影機圖像的線程函數"""
        while self.running:
            try:
                with self.top_camera_lock:
                    if not self.top_camera_connected or self.top_camera_conn is None:
                        time.sleep(0.1)
                        continue
                
                image = self._receive_image_internal(self.top_camera_conn)
                if image is not None:
                    try:
                        # 將圖像放入隊列，如果隊列已滿則丟棄最舊的圖像
                        if self.top_image_queue.full():
                            try:
                                self.top_image_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.top_image_queue.put_nowait(image)
                    except Exception as e:
                        print(f"將頂部攝影機圖像放入隊列時發生錯誤: {e}")
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
            length_bytes = self.recv_all(self.info_conn, 4)
            if not length_bytes:
                return None

            length = int.from_bytes(length_bytes, byteorder='big', signed=True)
            
            if length <= 0 or length > 10**6:
                print(f"接收到無效的資料長度: {length}")
                return None

            data_bytes = self.recv_all(self.info_conn, length)
            if not data_bytes:
                return None
            
            try:
                data_str = data_bytes.decode('utf-8')
                return json.loads(data_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失敗: {e}")
                return None

        except Exception as e:
            print(f"接收資訊數據時發生錯誤: {e}")
            return None
    
    def _receive_image_internal(self, conn: socket.socket) -> Optional[np.ndarray]:
        """內部方法：接收圖像數據"""
        try:
            image_len_bytes = self.recv_all(conn, 4)
            if not image_len_bytes:
                return None

            image_len = int.from_bytes(image_len_bytes, byteorder='little')
            image_data = self.recv_all(conn, image_len)
            if not image_data:
                return None

            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image

        except Exception as e:
            print(f"接收圖像數據時發生錯誤: {e}")
            return None
    
    async def setup_reset_server_async(self):
        """非同步設置重置伺服器"""
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.reset_socket.setblocking(False)
        
        try:
            self.reset_socket.bind(self.reset_address)
            self.reset_socket.listen(5)
            print("重置訊號伺服器已啟動，等待連接...")
            
            # 啟動重置連接處理線程
            self.reset_thread = threading.Thread(target=self.accept_reset_connections)
            self.reset_thread.daemon = True
            self.reset_thread.start()
            
            # 標記為已連接
            self.reset_connected = True
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")
    
    async def setup_top_camera_server(self):
        """設置頂部攝影機伺服器"""
        self.top_camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.top_camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.top_camera_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.top_camera_socket.setblocking(False)
        
        try:
            self.top_camera_socket.bind(self.top_camera_address)
            self.top_camera_socket.listen(5)
            print("頂部攝影機伺服器已啟動，等待連接...")
            
            conn, addr = await self.loop.sock_accept(self.top_camera_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            
            self.top_camera_conn = conn
            self.top_camera_connected = True
            print("已連接到頂部攝影機:", addr)
        except Exception as e:
            raise Exception(f"頂部攝影機伺服器設置失敗: {e}")

    async def setup_control_server(self):
        """設置控制伺服器"""
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.control_socket.setblocking(False)
        
        try:
            self.control_socket.bind(self.control_address)
            self.control_socket.listen(5)
            print('控制伺服器已啟動，等待連接...')
            
            conn, addr = await self.loop.sock_accept(self.control_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            
            self.control_conn = conn
            print("已連接到控制伺服器:", addr)
        except Exception as e:
            raise Exception(f"控制伺服器設置失敗: {e}")

    async def setup_info_server(self):
        """設置資訊伺服器"""
        self.info_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.info_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.info_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info_socket.setblocking(False)
        
        try:
            self.info_socket.bind(self.info_address)
            self.info_socket.listen(5)
            print('資訊伺服器已啟動，等待連接...')
            
            conn, addr = await self.loop.sock_accept(self.info_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            # 增加 keep-alive 設置和更大的 buffer
            conn.settimeout(5)  # 增加超時時間到 5 秒
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, 60)
            conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, 10)
            conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, 5)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)  # 增加接收緩衝區
            
            self.info_conn = conn
            print("已連接到資訊伺服器:", addr)
        except Exception as e:
            raise Exception(f"資訊伺服器設置失敗: {e}")

    async def setup_obs_server(self):
        """設置觀察伺服器"""
        self.obs_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.obs_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.obs_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.obs_socket.setblocking(False)
        
        try:
            self.obs_socket.bind(self.obs_address)
            self.obs_socket.listen(5)
            print("影像接收伺服器已啟動，等待連接...")
            
            conn, addr = await self.loop.sock_accept(self.obs_socket)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            
            self.obs_conn = conn
            print("已連接到影像接收伺服器:", addr)
        except Exception as e:
            raise Exception(f"影像接收伺服器設置失敗: {e}")

    def setup_reset_server(self):
        """設置重置伺服器"""
        self.reset_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reset_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.reset_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        try:
            self.reset_socket.bind(self.reset_address)
            self.reset_socket.listen(5)
            print("重置訊號伺服器已啟動，等待連接...")
            self.reset_socket.settimeout(10)
            
            self.reset_thread = threading.Thread(target=self.accept_reset_connections)
            self.reset_thread.daemon = True
            self.reset_thread.start()
        except Exception as e:
            raise Exception(f"重置伺服器設置失敗: {e}")

    def accept_reset_connections(self):
        """處理重置連接的線程函數"""
        connection_retry_delay = 0.5  # 連接重試延遲
        max_consecutive_errors = 5  # 最大連續錯誤次數
        consecutive_errors = 0  # 當前連續錯誤次數
        
        # 設置連接狀態為已連接，即使還沒有實際連接
        # 這樣可以避免Unity端因為連接狀態檢查而失敗
        self.reset_connected = True
        
        while not self.reset_thread_stop.is_set():
            reset_conn = None
            try:
                print("等待重置連接...")
                self.reset_socket.settimeout(1)
                reset_conn, reset_addr = self.reset_socket.accept()
                reset_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                reset_conn.settimeout(5)  # 增加超時時間，避免過早斷開連接
                print("已連接到重置訊號發送端:", reset_addr)
                
                # 重置連續錯誤計數
                consecutive_errors = 0
                
                # 設置連接狀態
                self.reset_conn = reset_conn
                
                while not self.reset_thread_stop.is_set():
                    try:
                        signal_data = reset_conn.recv(4)
                        if not signal_data:
                            print("連接已關閉，等待重新連接")
                            break
                            
                        if len(signal_data) != 4:
                            print(f"接收到不完整的重置信號數據: {len(signal_data)}/4 字節，繼續等待")
                            continue
                            
                        signal = int.from_bytes(signal_data, byteorder='little')
                        
                        if signal == 1:
                            # 增加 epoch 值
                            self.epoch += 1
                            epoch_data = self.epoch.to_bytes(4, byteorder='little')
                            
                            try:
                                # 確保數據完全發送
                                bytes_sent = reset_conn.send(epoch_data)
                                if bytes_sent != 4:
                                    print(f"警告: 只發送了 {bytes_sent}/4 字節的 epoch 數據")
                                    # 嘗試發送剩餘的數據
                                    remaining_data = epoch_data[bytes_sent:]
                                    while remaining_data:
                                        more_sent = reset_conn.send(remaining_data)
                                        if more_sent == 0:
                                            raise ConnectionError("無法發送剩餘數據")
                                        remaining_data = remaining_data[more_sent:]
                                
                                print(f"向 Unity 發送當前 epoch: {self.epoch}")
                            except Exception as e:
                                print(f"發送 epoch 到 Unity 時發生錯誤: {e}")
                                break
                            
                            # 設置重置事件
                            self.reset_event.set()
                            print(f"收到重置訊號，當前 epoch: {self.epoch}")
                    except socket.timeout:
                        # 超時只是表示沒有數據，不是錯誤
                        continue
                    except ConnectionResetError:
                        print("重置連接被遠端重置")
                        break
                    except ConnectionAbortedError:
                        print("重置連接被中止")
                        break
                    except Exception as e:
                        print(f"接收重置訊號時發生錯誤: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"連續錯誤次數達到 {max_consecutive_errors}，關閉連接")
                            break
                        continue
                
                # 關閉連接
                if reset_conn:
                    try:
                        reset_conn.shutdown(socket.SHUT_RDWR)
                    except:
                        pass
                    try:
                        reset_conn.close()
                    except:
                        pass
                
                # 重置連接，但保持連接狀態為True
                self.reset_conn = None
                print("重置連接已關閉，等待新連接...")
                
                # 短暫延遲，避免立即重新連接
                time.sleep(connection_retry_delay)
                
            except socket.timeout:
                # 接受連接超時，這是正常的
                continue
            except Exception as e:
                if not self.reset_thread_stop.is_set():
                    print(f"重置連接發生錯誤: {e}")
                    consecutive_errors += 1
                    
                    # 如果連續錯誤太多，增加延遲
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"連續錯誤次數達到 {max_consecutive_errors}，增加延遲")
                        time.sleep(connection_retry_delay * 2)
                        consecutive_errors = 0
                continue
            finally:
                # 確保連接被關閉，但保持連接狀態為True
                if reset_conn:
                    try:
                        reset_conn.close()
                    except:
                        pass

    def send_control_signal(self, relative_angle: float) -> None:
        """發送控制信號到Unity"""
        try:
            buffer = struct.pack('f', relative_angle)
            self.control_conn.sendall(buffer)
        except Exception as e:
            print(f"發送控制訊號時發生錯誤: {e}")
            self.control_conn, _ = self.reconnect_socket(
                self.control_socket, 
                self.control_address, 
                '控制'
            )

    def receive_image(self, show: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """接收影像數據 (使用緩存)"""
        with self.image_lock:
            image = self.image_cache
        
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
        
        return info

    def reconnect_socket(
        self, 
        socket_obj: socket.socket, 
        address: Tuple[str, int], 
        conn_type: str
    ) -> Tuple[Optional[socket.socket], Optional[Tuple[str, int]]]:
        """重新連接socket"""
        max_reconnect_attempts = 3
        reconnect_delay = 1.0
        
        for attempt in range(max_reconnect_attempts):
            try:
                socket_obj.settimeout(5)  # 設置接受連接的超時時間
                new_conn, new_addr = socket_obj.accept()
                new_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # 為資訊伺服器設置特殊參數
                if conn_type == '資訊':
                    new_conn.settimeout(5)
                    new_conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    new_conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, 60)
                    new_conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, 10)
                    new_conn.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, 5)
                    new_conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
                
                print(f"重新連接到{conn_type}伺服器 (嘗試 {attempt + 1}/{max_reconnect_attempts}):", new_addr)
                return new_conn, new_addr
            except socket.timeout:
                if attempt < max_reconnect_attempts - 1:
                    print(f"重新連接超時，等待 {reconnect_delay} 秒後重試...")
                    time.sleep(reconnect_delay)
                continue
            except Exception as e:
                print(f"重新連接{conn_type}伺服器時發生錯誤: {e}")
                if attempt < max_reconnect_attempts - 1:
                    print(f"等待 {reconnect_delay} 秒後重試...")
                    time.sleep(reconnect_delay)
                continue
        
        print(f"重新連接{conn_type}伺服器失敗，已達最大重試次數")
        return None, None

    def recv_all(self, conn: socket.socket, length: int) -> Optional[bytearray]:
        """接收指定長度的所有數據"""
        data = bytearray()
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def set_epoch(self, epoch: int) -> None:
        """設置當前世代"""
        self.epoch = epoch

    def is_reset_triggered(self) -> bool:
        """檢查是否觸發重置"""
        return self.reset_event.is_set()

    def clear_reset_event(self) -> None:
        """清除重置事件"""
        self.reset_event.clear()

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
            
            if hasattr(self, 'reset_thread') and self.reset_thread is not None:
                self.reset_thread.join(timeout=1)
        except Exception as e:
            print(f"關閉重置線程時發生錯誤: {e}")
            
        try:
            # 關閉所有連接和套接字
            connections = [
                ('control_conn', 'control_socket'),
                ('info_conn', 'info_socket'),
                ('obs_conn', 'obs_socket'),
                ('reset_conn', 'reset_socket'),
                ('top_camera_conn', 'top_camera_socket')
            ]
            
            for conn_attr, socket_attr in connections:
                # 關閉連接
                if hasattr(self, conn_attr) and getattr(self, conn_attr) is not None:
                    try:
                        conn = getattr(self, conn_attr)
                        conn.shutdown(socket.SHUT_RDWR)
                        conn.close()
                    except Exception as e:
                        print(f"關閉 {conn_attr} 時發生錯誤: {e}")
                    finally:
                        setattr(self, conn_attr, None)
                
                # 關閉套接字
                if hasattr(self, socket_attr) and getattr(self, socket_attr) is not None:
                    try:
                        sock = getattr(self, socket_attr)
                        sock.shutdown(socket.SHUT_RDWR)
                        sock.close()
                    except Exception as e:
                        print(f"關閉 {socket_attr} 時發生錯誤: {e}")
                    finally:
                        setattr(self, socket_attr, None)
        except Exception as e:
            print(f"關閉網路連接時發生錯誤: {e}")
            
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
