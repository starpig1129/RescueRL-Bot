import sys
import os
import time
from datetime import datetime
from tabulate import tabulate
from colorama import init, Fore, Style

class TrainLog:
    def __init__(self):
        # 初始化 colorama
        init()
        
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_steps': 1000000,
            'fps': 0.0,
            'time_elapsed': 0.0,
            'estimated_time': 0.0,
            'mean_reward': 0.0,
            'progress_bar': '',
            'person_detection': 0.0,
            'distance_reward': 0.0,
            'distance_penalty': 0.0,
            'inview_reward': 0.0,
            'viewdist_reward': 0.0,
            'viewdist_penalty': 0.0,
            'lost_view_penalty': 0.0,
            'movement_reward': 0.0,
            'movement_penalty': 0.0,
            'upside_down_penalty': 0.0,
            'touch_reward': 0.0,
            'continuous_reward': 0.0,
        }
        
        self.data_handler_stats = {
            'current_epoch_data': 0,
            'total_data_saved': 0,
            'feature_data_saved': 0,
            'data_save_rate': 0.0
        }
        
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_step = 0
        
        self.error_log_path = "error_logs"
        os.makedirs(self.error_log_path, exist_ok=True)

    def update_data_handler_stats(self, stats):
        """更新數據處理器統計資訊"""
        if stats is None:
            return
            
        self.data_handler_stats.update({
            'current_epoch_data': int(stats.get('current_epoch_data', 0)),
            'total_data_saved': int(stats.get('total_data_saved', 0)),
            'feature_data_saved': int(stats.get('total_features_saved', 0)),
            'data_save_rate': float(stats.get('data_save_rate', 0.0))
        })
        
    def _create_progress_bar(self, progress, width=50):
        """創建進度條"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = min(100, int(progress * 100))
        return f"{percentage}% {bar}"


    def _calculate_progress(self):
        if self.training_stats['total_steps'] > 0:
            return min(1.0, self.training_stats['step'] / self.training_stats['total_steps'])
        return 0

    def update_training_info(self, info):
        """更新訓練資訊"""
        if info is None:
            return
            
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
    
    def update_env_info(self, epoch, step, reward_list):
        """更新來自環境的資訊"""
        if epoch is not None:
            self.training_stats['epoch'] = epoch
        if step is not None:
            self.training_stats['step'] = step
        
        if reward_list is not None:
            reward_names = [
                'person_detection', 'distance_reward', 'distance_penalty',
                'inview_reward', 'viewdist_reward', 'viewdist_penalty',
                'lost_view_penalty', 'movement_reward', 'movement_penalty',
                'upside_down_penalty', 'touch_reward', 'continuous_reward'
            ]
            for name, value in zip(reward_names, reward_list):
                self.training_stats[name] = value



    def _calculate_time_stats(self):
        """計算時間統計資訊"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.training_stats['time_elapsed'] = elapsed_time
        
        # 計算預估剩餘時間
        if self.training_stats['fps'] > 0:
            current_step = self.training_stats['step']
            total_steps = self.training_stats['total_steps']
            remaining_steps = max(0, total_steps - current_step)
            
            # 使用當前 FPS 估算剩餘時間
            estimated_time = remaining_steps / max(self.training_stats['fps'], 1e-6)
            self.training_stats['estimated_time'] = max(0, estimated_time)
        
        # 計算進度和進度條
        progress = self._calculate_progress()
        self.training_stats['progress_bar'] = self._create_progress_bar(progress)
    def _format_time(self, seconds):
        """格式化時間顯示"""
        if seconds < 0 or not isinstance(seconds, (int, float)):
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _colored_value(self, value, format_spec=".2f"):
        """為數值添加顏色"""
        try:
            # 確保value是數值型別
            value = float(value)
            
            # 格式化數值
            if isinstance(format_spec, str):
                formatted_value = f"{value:{format_spec}}"
            else:
                formatted_value = f"{value:.2f}"
            
            # 添加顏色
            if value > 0:
                return f"{Fore.GREEN}{formatted_value}{Style.RESET_ALL}"
            elif value < 0:
                return f"{Fore.RED}{formatted_value}{Style.RESET_ALL}"
            return f"{formatted_value}"
        except (ValueError, TypeError):
            return "0.00"

    def display(self):
        """顯示訓練資訊"""
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
             f"當前世代數據量: {self.data_handler_stats['current_epoch_data']:,}",
             f"總儲存數據量: {self.data_handler_stats['total_data_saved']:,}",
             f"特徵數據量: {self.data_handler_stats['feature_data_saved']:,}",
             f"儲存速率: {self.data_handler_stats['data_save_rate']:.1f} 筆/秒"]
        ]
        
        # 計算並顯示進度條
        progress = min(1.0, self.training_stats['step'] / max(1, self.training_stats['total_steps']))
        bar_width = 50
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        progress_percentage = min(100, int(progress * 100))
        progress_bar = [[f"整體進度 {progress_percentage}%", bar]]
        
        # 格式化獎勵資訊
        reward_info = [
            ["獎勵詳細資訊 (1)",
             f"人物偵測: {self._colored_value(self.training_stats['person_detection'])}",
             f"距離增加: {self._colored_value(self.training_stats['distance_reward'])}",
             f"距離減少: {self._colored_value(self.training_stats['distance_penalty'])}",
             f"視野內: {self._colored_value(self.training_stats['inview_reward'])}",
             f"視距增加: {self._colored_value(self.training_stats['viewdist_reward'])}",
             f"視距減少: {self._colored_value(self.training_stats['viewdist_penalty'])}"]
        ]
        
        reward_info2 = [
            ["獎勵詳細資訊 (2)",
             f"失去視野: {self._colored_value(self.training_stats['lost_view_penalty'])}",
             f"移動獎勵: {self._colored_value(self.training_stats['movement_reward'])}",
             f"移動懲罰: {self._colored_value(self.training_stats['movement_penalty'])}",
             f"翻倒懲罰: {self._colored_value(self.training_stats['upside_down_penalty'])}",
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

    def log_info(self, message: str):
        """記錄一般信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - INFO: {message}")
        
    def log_error(self, error):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = os.path.join(self.error_log_path, f"error_{timestamp}.log")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"時間: {timestamp}\n")
            f.write(f"錯誤: {str(error)}\n")
        
        print(f"\n發生錯誤! 詳細資訊已儲存至: {error_file}")