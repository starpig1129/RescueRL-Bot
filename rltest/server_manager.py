import socket
import json
import struct
import time
import threading
import asyncio
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any

class ServerManager:
    def __init__(self, logger=None):
        self.logger = logger
        self._init_socket_vars()
        
        # 非同步相關變量
        self.loop = None
        self.tasks = []  # 存放所有非同步任務
        
        # 重置事件相關設置
        self.reset_event = threading.Event()
        self.reset_thread = None
        self.reset_thread_stop = threading.Event()
        
        # 當前世代
        self.epoch = 0

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

    def setup_all_servers(self):
        """設置並啟動所有伺服器"""
        try:
            # 創建或獲取事件循環
            self.loop = asyncio.get_event_loop()
            if self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
            self.setup_reset_server()
            # 使用run_until_complete運行所有伺服器設置任務
            self.loop.run_until_complete(asyncio.gather(
                self.setup_control_server(),
                self.setup_info_server(),
                self.setup_obs_server(),
                self.setup_top_camera_server()
            ))
        except Exception as e:
            self.close()
            raise Exception(f"設置伺服器時發生錯誤: {e}")

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
            conn.settimeout(1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024)
            
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
        while not self.reset_thread_stop.is_set():
            try:
                self.reset_socket.settimeout(1)
                reset_conn, reset_addr = self.reset_socket.accept()
                print("已連接到重置訊號發送端:", reset_addr)
                
                while not self.reset_thread_stop.is_set():
                    try:
                        signal_data = reset_conn.recv(4)
                        if not signal_data:
                            break
                        signal = int.from_bytes(signal_data, byteorder='little')
                        
                        if signal == 1:
                            epoch_data = self.epoch.to_bytes(4, byteorder='little')
                            try:
                                reset_conn.send(epoch_data)
                                print(f"向 Unity 發送當前 epoch: {self.epoch}")
                            except Exception as e:
                                print(f"發送 epoch 到 Unity 時發生錯誤: {e}")
                            
                            self.reset_event.set()
                            print(f"收到重置訊號，當前 epoch: {self.epoch}")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"接收重置訊號時發生錯誤: {e}")
                        break
                        
                reset_conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                if not self.reset_thread_stop.is_set():
                    print(f"重置連接發生錯誤: {e}")
                continue

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
        """接收影像數據"""
        try:
            image_len_bytes = self.recv_all(self.obs_conn, 4)
            if not image_len_bytes:
                return None, None

            image_len = int.from_bytes(image_len_bytes, byteorder='little')
            image_data = self.recv_all(self.obs_conn, image_len)
            if not image_data:
                return None, None

            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if show and image is not None:
                cv2.imshow("觀察空間", image)
                cv2.waitKey(1)
            
            return image, image.copy() if image is not None else None

        except Exception as e:
            print(f"接收影像資料時發生錯誤: {e}")
            self.obs_conn, _ = self.reconnect_socket(
                self.obs_socket, 
                self.obs_address, 
                '影像接收'
            )
            return None, None

    def receive_top_camera_image(self, show: bool = False) -> Optional[np.ndarray]:
        """接收頂部攝影機的影像"""
        try:
            image_len_bytes = self.recv_all(self.top_camera_conn, 4)
            if not image_len_bytes:
                return None

            image_len = int.from_bytes(image_len_bytes, byteorder='little')
            image_data = self.recv_all(self.top_camera_conn, image_len)
            if not image_data:
                return None

            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if show and image is not None:
                cv2.imshow("頂部視角", image)
                cv2.waitKey(1)
                
            return image

        except Exception as e:
            print(f"接收頂部攝影機影像時發生錯誤: {e}")
            self.top_camera_conn, _ = self.reconnect_socket(
                self.top_camera_socket,
                self.top_camera_address,
                '頂部攝影機'
            )
            return None

    def receive_info(self) -> Optional[Dict[str, Any]]:
        """接收資訊數據"""
        max_retries = 3
        retry_delay = 0.01
        
        for retry_count in range(max_retries):
            try:
                length_bytes = self.recv_all(self.info_conn, 4)
                if not length_bytes:
                    if retry_count < max_retries - 1:
                        print(f"未接收到資料長度 (重試 {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return None

                length = int.from_bytes(length_bytes, byteorder='big', signed=True)
                
                if length <= 0 or length > 10**6:
                    print(f"接收到無效的資料長度: {length}")
                    if retry_count < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

                data_bytes = self.recv_all(self.info_conn, length)
                if not data_bytes:
                    if retry_count < max_retries - 1:
                        print(f"未接收到完整資料 (重試 {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    return None
                
                try:
                    data_str = data_bytes.decode('utf-8')
                    return json.loads(data_str)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析失敗: {e}")
                    if retry_count < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

            except socket.timeout:
                if retry_count < max_retries - 1:
                    print(f"接收資料超時 (重試 {retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                return None
            except ConnectionResetError:
                print("連接被重置，嘗試重新連接...")
                self.info_conn, _ = self.reconnect_socket(
                    self.info_socket,
                    self.info_address,
                    '資訊'
                )
                if retry_count < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
            except Exception as e:
                print(f"接收資料時發生錯誤: {e}")
                if retry_count < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None
        
        return None

    def reconnect_socket(
        self, 
        socket_obj: socket.socket, 
        address: Tuple[str, int], 
        conn_type: str
    ) -> Tuple[Optional[socket.socket], Optional[Tuple[str, int]]]:
        """重新連接socket"""
        try:
            new_conn, new_addr = socket_obj.accept()
            new_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"重新連接到{conn_type}伺服器:", new_addr)
            return new_conn, new_addr
        except Exception as e:
            print(f"重新連接{conn_type}伺服器時發生錯誤: {e}")
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
