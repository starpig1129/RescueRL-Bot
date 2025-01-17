import sys
import os
import time
from datetime import datetime
from tabulate import tabulate
from colorama import init, Fore, Style
from threading import Lock

class TrainLog:
    def __init__(self):
        """初始化訓練日誌系統"""
        # 初始化 colorama
        init()
        
        # 添加鎖以確保執行緒安全
        self.stats_lock = Lock()
        
        # 訓練相關統計資料
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_steps': 1000000,
            'fps': 0.0,
            'time_elapsed': 0.0,
            'estimated_time': 0.0,
            'mean_reward': 0.0,
            'progress_bar': '',
            'person_detection': 0.0,        # 人像偵測獎勵
            'approach_reward': 0.0,         # 接近目標獎勵
            'leave_penalty': 0.0,           # 遠離目標懲罰
            'inview_reward': 0.0,           # 目標在視野內獎勵
            'center_approach': 0.0,         # 目標接近視野中心獎勵
            'center_leave': 0.0,            # 目標遠離視野中心懲罰
            'lost_view': 0.0,               # 失去目標視野懲罰
            'movement_reward': 0.0,         # 移動獎勵
            'movement_penalty': 0.0,        # 移動距離懲罰
            'posture_penalty': 0.0,         # 姿態偏差懲罰
            'touch_reward': 0.0,            # 碰觸目標獎勵
            'continuous_reward': 0.0,       # 持續碰觸獎勵
        }
        
        # 數據處理相關統計資料
        self.data_stats = {
            'current_epoch_data': 0,
            'total_data_saved': 0,
            'feature_data_saved': 0,
            'data_save_rate': 0.0,
            'write_queue_size': 0,
            'disk_usage': 0.0,
            'resize_count': 0,
            'max_steps': 0,
            'max_feature_steps': 0
        }
        
        # 時間相關變數
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_step = 0
        
        # 日誌檔案路徑
        self.log_base_path = "logs"
        self.error_log_path = os.path.join(self.log_base_path, "errors")
        self.info_log_path = os.path.join(self.log_base_path, "info")
        
        # 建立日誌目錄
        os.makedirs(self.error_log_path, exist_ok=True)
        os.makedirs(self.info_log_path, exist_ok=True)

    def log_info(self, message: str) -> None:
        """
        記錄一般資訊
        
        參數:
            message: 要記錄的訊息
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - INFO: {message}"
            
            # 打印到控制台
            print(log_message)
            
            # 記錄到檔案
            current_date = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(self.info_log_path, f"info_{current_date}.log")
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + "\n")
                
        except Exception as e:
            print(f"記錄資訊時發生錯誤: {str(e)}")
            self.log_error(e)
    
    def update_data_handler_stats(self, stats: dict) -> None:
        """
        更新數據處理器統計資訊
        
        參數:
            stats: 包含統計資訊的字典
        """
        if stats is None:
            return
            
        with self.stats_lock:
            try:
                self.data_stats.update({
                    'current_epoch_data': int(stats.get('current_epoch_data', 0)),
                    'total_data_saved': int(stats.get('total_data_saved', 0)),
                    'feature_data_saved': int(stats.get('total_features_saved', 0)),
                    'data_save_rate': float(stats.get('data_save_rate', 0.0)),
                    'write_queue_size': int(stats.get('write_queue_size', 0)),
                    'disk_usage': float(stats.get('disk_usage', 0.0)),
                    'resize_count': int(stats.get('total_resize_count', 0)),
                    'max_steps': int(stats.get('max_steps', 0)),
                    'max_feature_steps': int(stats.get('max_feature_steps', 0))
                })
            except Exception as e:
                self.log_error(f"更新數據處理器統計時發生錯誤: {str(e)}")

    def update_training_info(self, info: dict) -> None:
        """
        更新訓練資訊
        
        參數:
            info: 包含訓練資訊的字典
        """
        if info is None:
            return
            
        with self.stats_lock:
            try:
                # 更新基本訓練資訊
                self.training_stats.update({
                    'fps': float(info.get('fps', 0.0)),
                    'step': int(info.get('step', 0)),
                    'total_steps': int(info.get('max_steps', 1000000)),
                    'mean_reward': float(info.get('mean_reward', 0.0))
                })
                
                # 計算時間相關統計
                current_time = time.time()
                self.training_stats['time_elapsed'] = current_time - self.start_time
                
                # 計算預估剩餘時間
                if self.training_stats['fps'] > 0:
                    remaining_steps = max(0, self.training_stats['total_steps'] - self.training_stats['step'])
                    self.training_stats['estimated_time'] = remaining_steps / max(self.training_stats['fps'], 1e-6)
            except Exception as e:
                self.log_error(f"更新訓練資訊時發生錯誤: {str(e)}")

    def update_env_info(self, epoch: int, step: int, reward_list: list) -> None:
        """
        更新環境資訊
        
        參數:
            epoch: 當前世代
            step: 當前步數
            reward_list: 獎勵列表
        """
        with self.stats_lock:
            try:
                if epoch is not None:
                    self.training_stats['epoch'] = epoch
                if step is not None:
                    self.training_stats['step'] = step
                
                if reward_list is not None:
                    reward_names = [
                        'person_detection',   # 人像偵測獎勵
                        'approach_reward',    # 接近目標獎勵
                        'leave_penalty',      # 遠離目標懲罰
                        'inview_reward',      # 目標在視野內獎勵
                        'center_approach',    # 目標接近視野中心獎勵
                        'center_leave',       # 目標遠離視野中心懲罰
                        'lost_view',          # 失去目標視野懲罰
                        'movement_reward',    # 移動獎勵
                        'movement_penalty',   # 移動距離懲罰
                        'posture_penalty',    # 姿態偏差懲罰
                        'touch_reward',       # 碰觸目標獎勵
                        'continuous_reward'   # 持續碰觸獎勵
                    ]
                    for name, value in zip(reward_names, reward_list):
                        self.training_stats[name] = value
            except Exception as e:
                self.log_error(f"更新環境資訊時發生錯誤: {str(e)}")

    def _create_progress_bar(self, progress: float, width: int = 50) -> str:
        """
        創建進度條
        
        參數:
            progress: 進度值 (0-1)
            width: 進度條寬度
            
        返回:
            格式化的進度條字串
        """
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = min(100, int(progress * 100))
        return f"{percentage}% {bar}"

    def _calculate_progress(self) -> float:
        """
        計算當前訓練進度
        
        返回:
            進度值 (0-1)
        """
        if self.training_stats['total_steps'] > 0:
            return min(1.0, self.training_stats['step'] / self.training_stats['total_steps'])
        return 0

    def display(self) -> None:
        """顯示訓練資訊"""
        with self.stats_lock:
            try:
                # 清除螢幕
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # 格式化進度資訊
                progress_info = [
                    ["訓練進度指標",
                     f"世代: {self.training_stats['epoch']}",
                     f"步數: {self.training_stats['step']}/{self.training_stats['total_steps']}",
                     f"速度: {self.training_stats['fps']:.1f} FPS",
                     f"已訓練: {self._format_time(self.training_stats['time_elapsed'])}",
                     f"預估剩餘: {self._format_time(self.training_stats['estimated_time'])}",
                     f"平均獎勵: {self._colored_value(self.training_stats['mean_reward'])}"]
                ]
                
                # 格式化數據統計
                data_stats = [
                    ["數據儲存統計",
                     f"當前世代數據量: {self.data_stats['current_epoch_data']:,d}",
                     f"總儲存數據量: {self.data_stats['total_data_saved']:,d}",
                     f"特徵數據量: {self.data_stats['feature_data_saved']:,d}",
                     f"儲存速率: {self.data_stats['data_save_rate']:.2f} 筆/秒",
                     f"寫入隊列: {self.data_stats['write_queue_size']} 筆",
                     f"磁碟使用: {self.data_stats['disk_usage']:.2f} MB"]
                ]
                
                # 計算並顯示進度條
                progress = self._calculate_progress()
                bar = self._create_progress_bar(progress)
                progress_bar = [[f"整體進度 {int(progress * 100)}%", bar]]
                
                # 格式化獎勵資訊
                reward_info = [
                    ["獎勵詳細資訊 (1)",
                     f"人物偵測: {self._colored_value(self.training_stats['person_detection'])}",
                     f"接近目標: {self._colored_value(self.training_stats['approach_reward'])}",
                     f"遠離目標: {self._colored_value(self.training_stats['leave_penalty'])}",
                     f"在視野內: {self._colored_value(self.training_stats['inview_reward'])}",
                     f"接近中心: {self._colored_value(self.training_stats['center_approach'])}",
                     f"遠離中心: {self._colored_value(self.training_stats['center_leave'])}"]
                ]
                
                reward_info2 = [
                    ["獎勵詳細資訊 (2)",
                     f"失去視野: {self._colored_value(self.training_stats['lost_view'])}",
                     f"移動獎勵: {self._colored_value(self.training_stats['movement_reward'])}",
                     f"移動懲罰: {self._colored_value(self.training_stats['movement_penalty'])}",
                     f"姿態懲罰: {self._colored_value(self.training_stats['posture_penalty'])}",
                     f"觸碰獎勵: {self._colored_value(self.training_stats['touch_reward'])}",
                     f"持續獎勵: {self._colored_value(self.training_stats['continuous_reward'])}"]
                ]

                # 顯示所有資訊
                print("\n訓練狀態:")
                print(tabulate(progress_info, tablefmt="grid"))
                print("\n數據儲存狀態:")
                print(tabulate(data_stats, tablefmt="grid"))
                print("\n訓練進度:")
                print(tabulate(progress_bar, tablefmt="grid"))
                print("\n獎勵明細:")
                print(tabulate(reward_info, tablefmt="grid"))
                print(tabulate(reward_info2, tablefmt="grid"))
                
            except Exception as e:
                self.log_error(f"顯示資訊時發生錯誤: {str(e)}")

    def _format_time(self, seconds: float) -> str:
        """
        格式化時間顯示
        
        參數:
            seconds: 秒數
            
        返回:
            格式化的時間字串 (HH:MM:SS)
        """
        try:
            if seconds < 0 or not isinstance(seconds, (int, float)):
                return "00:00:00"
            
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            return "00:00:00"

    def _colored_value(self, value: float, format_spec: str = "12.2f") -> str:
        """
        為數值添加顏色
        
        參數:
            value: 要著色的數值
            format_spec: 格式化規格
            
        返回:
            著色後的字串
        """
        try:
            value = float(value)
            # 使用固定寬度格式化,確保有足夠空間顯示大數值
            formatted_value = f"{value:{format_spec}}" if isinstance(format_spec, str) else f"{value:12.2f}"
            
            if value > 0:
                return f"{Fore.GREEN}{formatted_value}{Style.RESET_ALL}"
            elif value < 0:
                return f"{Fore.RED}{formatted_value}{Style.RESET_ALL}"
            return formatted_value
        except (ValueError, TypeError):
            return "       0.00"  # 保持與其他數值對齊

    def log_error(self, error: Exception) -> None:
        """
        記錄錯誤到檔案
        
        參數:
            error: 錯誤物件或錯誤訊息
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_date = datetime.now().strftime("%Y%m%d")
            error_file = os.path.join(self.error_log_path, f"error_{current_date}.log")
            
            error_message = f"\n時間: {timestamp}\n"
            error_message += f"錯誤: {str(error)}\n"
            
            # 如果是異常物件，添加堆疊追蹤
            if isinstance(error, Exception):
                import traceback
                error_message += "堆疊追蹤:\n"
                error_message += "".join(traceback.format_tb(error.__traceback__))
            
            # 寫入錯誤日誌
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_message)
                f.write("-" * 80 + "\n")
            
            # 在控制台顯示錯誤訊息
            print(f"\n{Fore.RED}發生錯誤！{Style.RESET_ALL}")
            print(f"詳細資訊已儲存至: {error_file}")
            print(f"錯誤訊息: {str(error)}\n")
            
        except Exception as e:
            print(f"{Fore.RED}記錄錯誤時發生異常: {str(e)}{Style.RESET_ALL}")
            print(f"原始錯誤: {str(error)}")

    def get_training_stats(self) -> dict:
        """
        獲取當前訓練統計資訊的複本
        
        返回:
            訓練統計資訊的字典複本
        """
        with self.stats_lock:
            return self.training_stats.copy()

    def get_data_stats(self) -> dict:
        """
        獲取當前數據統計資訊的複本
        
        返回:
            數據統計資訊的字典複本
        """
        with self.stats_lock:
            return self.data_stats.copy()

    def reset_stats(self) -> None:
        """重置所有統計資訊"""
        with self.stats_lock:
            # 重置訓練統計
            for key in self.training_stats:
                if isinstance(self.training_stats[key], (int, float)):
                    self.training_stats[key] = 0
            
            # 重置數據統計
            for key in self.data_stats:
                if isinstance(self.data_stats[key], (int, float)):
                    self.data_stats[key] = 0
            
            # 重置時間相關變數
            self.start_time = time.time()
            self.last_update_time = time.time()
            self.last_step = 0

    def cleanup(self) -> None:
        """清理舊的日誌檔案（保留最近7天的日誌）"""
        try:
            current_time = time.time()
            max_age = 7 * 24 * 60 * 60  # 7天的秒數
            
            # 清理資訊日誌
            self._cleanup_directory(self.info_log_path, max_age)
            # 清理錯誤日誌
            self._cleanup_directory(self.error_log_path, max_age)
            
        except Exception as e:
            self.log_error(f"清理日誌檔案時發生錯誤: {str(e)}")

    def _cleanup_directory(self, directory: str, max_age: int) -> None:
        """
        清理指定目錄中的舊檔案
        
        參數:
            directory: 目錄路徑
            max_age: 檔案最大保留時間（秒）
        """
        current_time = time.time()
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            try:
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if current_time - file_time > max_age:
                        os.remove(filepath)
            except Exception as e:
                self.log_error(f"清理檔案 {filepath} 時發生錯誤: {str(e)}")

    def __del__(self):
        """解構子：確保在物件被銷毀時執行清理工作"""
        try:
            self.cleanup()
        except:
            pass
