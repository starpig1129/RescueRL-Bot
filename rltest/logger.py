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
            'fps': 0,
            'time_elapsed': 0,
            'estimated_time': 0,
            'mean_reward': 0,
            'progress_bar': '',
            'person_detection': 0,
            'distance_reward': 0,
            'distance_penalty': 0,
            'inview_reward': 0,
            'viewdist_reward': 0,
            'viewdist_penalty': 0,
            'lost_view_penalty': 0,
            'movement_reward': 0,
            'movement_penalty': 0,
            'upside_down_penalty': 0,
            'touch_reward': 0,
            'continuous_reward': 0,
        }
        
        # 新增數據處理器統計
        self.data_handler_stats = {
            'current_epoch_data': 0,    # 當前世代數據量
            'total_data_saved': 0,      # 總保存數據量
            'feature_data_saved': 0,    # 特徵數據量
            'data_save_rate': 0.0       # 保存速率
        }
        
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.last_step = 0
        
        self.error_log_path = "error_logs"
        os.makedirs(self.error_log_path, exist_ok=True)

    def update_data_handler_stats(self, stats: dict):
        """更新數據處理器統計信息"""
        self.data_handler_stats.update({
            'current_epoch_data': stats.get('current_epoch_data', 0),
            'total_data_saved': stats.get('total_data_saved', 0),
            'feature_data_saved': stats.get('current_epoch_features', 0),
            'data_save_rate': stats.get('data_save_rate', 0.0)
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

    def update_training_info(self, stable_baselines_info):
        """更新來自 StableBaselines3 的訓練資訊"""
        if stable_baselines_info is not None:
            # 更新 FPS
            if 'fps' in stable_baselines_info:
                self.training_stats['fps'] = stable_baselines_info['fps']
            
            # 更新步數資訊
            if 'step' in stable_baselines_info:
                self.training_stats['step'] = stable_baselines_info['step']
            if 'max_steps' in stable_baselines_info:
                self.training_stats['total_steps'] = stable_baselines_info['max_steps']
            
            # 更新平均獎勵
            if 'mean_reward' in stable_baselines_info:
                self.training_stats['mean_reward'] = stable_baselines_info['mean_reward']
    
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
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _colored_value(self, value, format_spec=".1f"):
        """為數值添加顏色
        Args:
            value: 要格式化的數值
            format_spec: 格式化規範，預設保留一位小數
        Returns:
            str: 帶顏色的格式化字串
        """
        try:
            # 先格式化數值
            if isinstance(format_spec, str):
                formatted_value = f"{float(value):{format_spec}}"
            else:
                formatted_value = f"{float(value):.1f}"
                
            # 根據數值添加顏色
            if value > 0:
                return f"{Fore.GREEN}{formatted_value}{Style.RESET_ALL}"
            elif value < 0:
                return f"{Fore.RED}{formatted_value}{Style.RESET_ALL}"
            return formatted_value
        except (ValueError, TypeError):
            # 如果無法格式化，返回原始值的字串形式
            return str(value)

    def display(self):
        """顯示訓練資訊"""
        self._calculate_time_stats()
        
        # 格式化步數
        current_step = min(self.training_stats['step'], self.training_stats['total_steps'])
        step_display = f"{current_step}/{self.training_stats['total_steps']}"
        
        # 進度資訊
        progress_info = [
            ["進度指標",
             f"世代: {self.training_stats['epoch']}",
             f"步數: {step_display}",
             f"速度: {self.training_stats['fps']:.1f} FPS",
             f"已訓練: {self._format_time(self.training_stats['time_elapsed'])}",
             f"預估剩餘: {self._format_time(self.training_stats['estimated_time'])}",
             f"平均獎勵: {self._colored_value(self.training_stats['mean_reward'], '.2f')}"]
        ]
        
        # 數據保存狀態
        data_stats = [
            ["數據統計",
             f"當前世代數據: {self.data_handler_stats['current_epoch_data']}",
             f"總保存數據: {self.data_handler_stats['total_data_saved']}",
             f"特徵數據: {self.data_handler_stats['feature_data_saved']}",
             f"保存速率: {self.data_handler_stats['data_save_rate']:.1f}/s"]
        ]
        
        # 進度條
        if self.training_stats['step'] < self.training_stats['total_steps']:
            progress_bar = [["訓練進度", self.training_stats['progress_bar']]]
        else:
            progress_bar = [["訓練進度", "100% " + "█" * 50]]
        
        # 獎勵資訊
        reward_info = [
            ["獎勵明細",
             f"偵測: {self._colored_value(self.training_stats['person_detection'])}",
             f"距離+: {self._colored_value(self.training_stats['distance_reward'])}",
             f"距離-: {self._colored_value(self.training_stats['distance_penalty'])}",
             f"視野: {self._colored_value(self.training_stats['inview_reward'])}",
             f"視距+: {self._colored_value(self.training_stats['viewdist_reward'])}",
             f"視距-: {self._colored_value(self.training_stats['viewdist_penalty'])}"]
        ]
        
        reward_info2 = [
            ["獎勵明細2",
             f"失視野: {self._colored_value(self.training_stats['lost_view_penalty'])}",
             f"移動+: {self._colored_value(self.training_stats['movement_reward'])}",
             f"移動-: {self._colored_value(self.training_stats['movement_penalty'])}",
             f"翻倒: {self._colored_value(self.training_stats['upside_down_penalty'])}",
             f"碰觸: {self._colored_value(self.training_stats['touch_reward'])}",
             f"持續: {self._colored_value(self.training_stats['continuous_reward'])}"]
        ]

        # 清除螢幕
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')

        # 顯示數據保存狀態
        print("\n數據保存狀態:")
        print(tabulate(data_stats, tablefmt="grid"))
        
        # 顯示訓練狀態
        print("\n訓練狀態:")
        print(tabulate(progress_info, tablefmt="grid"))
        print(tabulate(progress_bar, tablefmt="grid"))
        
        # 顯示獎勵詳情
        print("\n獎勵詳情:")
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