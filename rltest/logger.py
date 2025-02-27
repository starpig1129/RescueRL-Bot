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
        try:
            # 初始化 colorama
            init()
            
            # 添加鎖以確保執行緒安全
            self.stats_lock = Lock()
            
            # 獲取當前腳本的目錄作為基礎路徑
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 日誌檔案路徑
            self.log_base_path = os.path.join(script_dir, "logs")
            self.error_log_path = os.path.join(self.log_base_path, "errors")
            self.info_log_path = os.path.join(self.log_base_path, "info")
            
            print(f"初始化日誌路徑:")
            print(f"基礎路徑: {os.path.abspath(self.log_base_path)}")
            print(f"錯誤日誌: {os.path.abspath(self.error_log_path)}")
            print(f"資訊日誌: {os.path.abspath(self.info_log_path)}")
            
            # 建立日誌目錄
            print(f"正在創建日誌目錄...")
            print(f"錯誤日誌路徑: {os.path.abspath(self.error_log_path)}")
            print(f"資訊日誌路徑: {os.path.abspath(self.info_log_path)}")
            os.makedirs(self.error_log_path, exist_ok=True)
            os.makedirs(self.info_log_path, exist_ok=True)
            print("日誌目錄創建完成")
            
            # 初始化統計相關變數
            self._init_stats()
            
        except Exception as e:
            print(f"初始化日誌系統時發生錯誤: {e}")
            raise

    def _init_stats(self):
        """初始化統計資料"""
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

    def _safe_file_write(self, filepath: str, content: str) -> bool:
        """安全地寫入文件"""
        abs_path = os.path.abspath(filepath)
        try:
            # 確保目錄存在
            dirpath = os.path.dirname(abs_path)
            os.makedirs(dirpath, exist_ok=True)
            
            # 如果檔案不存在，先創建檔案
            if not os.path.exists(abs_path):
                with open(abs_path, 'w', encoding='utf-8') as _:
                    pass
                
            print(f"正在寫入檔案: {abs_path}")
            with open(abs_path, 'a', encoding='utf-8') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
                print(f"寫入完成，內容: {content.strip()}")
                
            return os.path.exists(abs_path)
            
        except Exception as e:
            print(f"寫入檔案時發生錯誤: {e}")
            print(f"檔案路徑: {abs_path}")
            return False

    def log_info(self, message: str) -> None:
        """記錄一般資訊"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - INFO: {message}\n"
            
            # 打印到控制台
            print(f"準備記錄資訊: {log_message.rstrip()}")
            
            # 記錄到檔案
            current_date = datetime.now().strftime("%Y%m%d")
            log_file = os.path.join(self.info_log_path, f"info_{current_date}.log")
            print(f"日誌檔案路徑: {os.path.abspath(log_file)}")
            
            # 寫入並確認
            if self._safe_file_write(log_file, log_message):
                print("資訊已成功記錄到檔案")
            else:
                print("資訊記錄失敗")
                
        except Exception as e:
            print(f"記錄資訊時發生錯誤: {e}")
            self.log_error(e)

    def _acquire_lock_with_timeout(self, timeout=1.0):
        """安全地獲取鎖，避免死鎖"""
        start_time = time.time()
        while True:
            if self.stats_lock.acquire(blocking=False):
                return True
            if time.time() - start_time > timeout:
                print("警告：無法獲取統計資料鎖，可能存在死鎖風險")
                return False
            time.sleep(0.1)

    def _release_lock_safe(self):
        """安全地釋放鎖"""
        try:
            self.stats_lock.release()
        except RuntimeError:
            pass

    def update_data_handler_stats(self, stats: dict) -> None:
        """更新數據處理器統計資訊"""
        if stats is None:
            return
            
        if not self._acquire_lock_with_timeout():
            return
            
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
            self.log_error(f"更新數據處理器統計時發生錯誤: {e}")
        finally:
            self._release_lock_safe()

    def update_training_info(self, info: dict) -> None:
        """更新訓練資訊"""
        if info is None:
            return
            
        if not self._acquire_lock_with_timeout():
            return
            
        try:
            self.training_stats.update({
                'fps': float(info.get('fps', 0.0)),
                'step': int(info.get('step', 0)),
                'total_steps': int(info.get('max_steps', 1000000)),
                'mean_reward': float(info.get('mean_reward', 0.0))
            })
            
            current_time = time.time()
            self.training_stats['time_elapsed'] = current_time - self.start_time
            
            if self.training_stats['fps'] > 0:
                remaining_steps = max(0, self.training_stats['total_steps'] - self.training_stats['step'])
                self.training_stats['estimated_time'] = remaining_steps / max(self.training_stats['fps'], 1e-6)
                
        except Exception as e:
            self.log_error(f"更新訓練資訊時發生錯誤: {e}")
        finally:
            self._release_lock_safe()

    def update_env_info(self, epoch: int, step: int, reward_list: list) -> None:
        """更新環境資訊"""
        if not self._acquire_lock_with_timeout():
            return
            
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
            self.log_error(f"更新環境資訊時發生錯誤: {e}")
        finally:
            self._release_lock_safe()

    def display(self) -> None:
        """顯示訓練資訊"""
        if not self._acquire_lock_with_timeout():
            return
            
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

            sys.stdout.write("\n訓練狀態:\n")
            sys.stdout.write(tabulate(progress_info, tablefmt="grid") + "\n")
            sys.stdout.write("\n數據儲存狀態:\n")
            sys.stdout.write(tabulate(data_stats, tablefmt="grid") + "\n")
            sys.stdout.write("\n訓練進度:\n")
            sys.stdout.write(tabulate(progress_bar, tablefmt="grid") + "\n")
            sys.stdout.write("\n獎勵明細:\n")
            sys.stdout.write(tabulate(reward_info, tablefmt="grid") + "\n")
            sys.stdout.write(tabulate(reward_info2, tablefmt="grid") + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            self.log_error(f"顯示資訊時發生錯誤: {e}")
        finally:
            self._release_lock_safe()

    def _create_progress_bar(self, progress: float, width: int = 50) -> str:
        """創建進度條"""
        try:
            filled = int(width * max(0, min(1, progress)))
            bar = '█' * filled + '░' * (width - filled)
            percentage = min(100, max(0, int(progress * 100)))
            return f"{percentage}% {bar}"
        except Exception:
            return "0% " + "░" * width

    def _calculate_progress(self) -> float:
        """計算當前訓練進度"""
        try:
            if self.training_stats['total_steps'] > 0:
                return min(1.0, max(0.0, self.training_stats['step'] / self.training_stats['total_steps']))
            return 0.0
        except Exception:
            return 0.0

    def _format_time(self, seconds: float) -> str:
        """格式化時間顯示"""
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
        """為數值添加顏色"""
        try:
            value = float(value)
            formatted_value = f"{value:{format_spec}}"
            
            if value > 0:
                return f"{Fore.GREEN}{formatted_value}{Style.RESET_ALL}"
            elif value < 0:
                return f"{Fore.RED}{formatted_value}{Style.RESET_ALL}"
            return formatted_value
        except (ValueError, TypeError):
            return "       0.00"

    def log_error(self, error: Exception) -> None:
        """記錄錯誤到檔案"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_date = datetime.now().strftime("%Y%m%d")
            error_file = os.path.join(self.error_log_path, f"error_{current_date}.log")
            print(f"\n準備寫入錯誤日誌: {os.path.abspath(error_file)}")
            
            error_message = f"\n時間: {timestamp}\n"
            error_message += f"錯誤: {str(error)}\n"
            
            if isinstance(error, Exception) and hasattr(error, '__traceback__'):
                import traceback
                error_message += "堆疊追蹤:\n"
                error_message += "".join(traceback.format_tb(error.__traceback__))
            
            error_message += "-" * 80 + "\n"
            
            # 寫入錯誤日誌
            write_success = self._safe_file_write(error_file, error_message)
            
            # 在控制台顯示錯誤訊息
            sys.stderr.write(f"\n{Fore.RED}發生錯誤！{Style.RESET_ALL}\n")
            if write_success:
                sys.stderr.write(f"錯誤日誌已寫入: {error_file}\n")
            else:
                sys.stderr.write(f"錯誤日誌寫入失敗!\n")
            sys.stderr.write(f"錯誤訊息: {str(error)}\n")
            sys.stderr.write(f"完整錯誤內容:\n{error_message}\n")
            sys.stderr.flush()
            
        except Exception as e:
            sys.stderr.write(f"{Fore.RED}記錄錯誤時發生異常: {str(e)}{Style.RESET_ALL}\n")
            sys.stderr.write(f"原始錯誤: {str(error)}\n")
            sys.stderr.flush()

    def _cleanup_directory(self, directory: str, max_age: int) -> None:
        """清理指定目錄中的舊檔案"""
        if not os.path.exists(directory):
            return

        current_time = time.time()
        try:
            for filename in os.listdir(directory):
                if filename.startswith('.'): # 跳過隱藏文件
                    continue
                    
                filepath = os.path.join(directory, filename)
                try:
                    if os.path.isfile(filepath):
                        if current_time - os.path.getmtime(filepath) > max_age:
                            os.remove(filepath)
                except PermissionError:
                    print(f"無法刪除檔案（權限不足）: {filepath}")
                except FileNotFoundError:
                    pass  # 檔案可能已被其他程序刪除
                except Exception as e:
                    self.log_error(f"清理檔案 {filepath} 時發生錯誤: {e}")
        except Exception as e:
            self.log_error(f"清理目錄 {directory} 時發生錯誤: {e}")

    def cleanup(self) -> None:
        """執行完整的清理工作"""
        try:
            # 確保所有待寫入的數據都已保存
            sys.stdout.flush()
            sys.stderr.flush()
            
            # 清理舊的日誌文件
            max_age = 7 * 24 * 60 * 60  # 7天的秒數
            self._cleanup_directory(self.info_log_path, max_age)
            self._cleanup_directory(self.error_log_path, max_age)
            
            # 重置統計數據
            self.reset_stats()
            
            # 釋放鎖
            if hasattr(self, 'stats_lock'):
                self._release_lock_safe()
                
        except Exception as e:
            print(f"清理時發生錯誤: {e}")

    def reset_stats(self) -> None:
        """重置所有統計資訊"""
        if not self._acquire_lock_with_timeout():
            return
            
        try:
            self._init_stats()
        except Exception as e:
            self.log_error(f"重置統計資訊時發生錯誤: {e}")
        finally:
            self._release_lock_safe()

    def __del__(self):
        """解構函數：確保資源被正確釋放"""
        try:
            # 進行清理
            self.cleanup()
            
            # 確保所有文件已關閉
            for key, value in vars(self).items():
                if hasattr(value, 'close'):
                    try:
                        value.close()
                    except:
                        pass
                        
            # 釋放資源
            self.training_stats = None
            self.data_stats = None
            
        except Exception as e:
            print(f"解構時發生錯誤: {e}")
