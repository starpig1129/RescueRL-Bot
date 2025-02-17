import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_rewards_from_epoch(data_dir, epoch, interval=1, memory_monitor=False, cache_dir=None):
    """只讀取一個世代的reward_list數據"""
    # 檢查緩存
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, f'reward_cache_ep{epoch:03d}_i{interval}.npy')
        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception as e:
                print(f"讀取緩存失敗: {e}")
    
    env_file_path = os.path.join(data_dir, "env_data", f"ep{epoch:03d}_env.h5")
    
    if not os.path.exists(env_file_path):
        return None
    
    try:
        with h5py.File(env_file_path, 'r') as env_file:
            # 讀取reward_list和對應的步數映射
            reward_steps = env_file['reward_step_map'][:]
            reward_list = env_file['reward_list'][:]
            
            # 根據步數映射進行採樣
            if interval > 1:
                # 使用reward_steps來確保採樣的時間點正確
                target_steps = np.arange(0, reward_steps[-1], interval)
                sample_indices = np.searchsorted(reward_steps, target_steps)
                sample_indices = sample_indices[sample_indices < len(reward_list)]
                reward_list = reward_list[sample_indices]
            
            if memory_monitor:
                memory_usage = reward_list.nbytes / (1024 * 1024)  # MB
                print(f"世代 {epoch} 的reward_list使用記憶體: {memory_usage:.2f} MB")
            
            # 儲存緩存
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_file, reward_list)
            
            return reward_list
    except Exception as e:
        print(f"讀取世代 {epoch} 時發生錯誤: {e}")
        return None

def plot_rewards(data_dir, start_epoch, end_epoch, interval=1, date_suffix=False, use_cache=True):
    """繪製獎勵變化曲線"""
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"正在讀取世代 {start_epoch} 到 {end_epoch} 的數據...")
    all_rewards = []
    
    # 設置緩存目錄
    cache_dir = "analysis_cache" if use_cache else None
    if use_cache:
        print("使用緩存加速數據讀取...")
    
    # 使用線程池並行讀取數據
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 創建讀取任務
        future_to_epoch = {
            executor.submit(
                load_rewards_from_epoch, 
                data_dir, epoch, interval, 
                memory_monitor=False, 
                cache_dir=cache_dir
            ): epoch
            for epoch in range(start_epoch, end_epoch + 1)
        }
        
        # 使用tqdm顯示進度
        for future in tqdm(future_to_epoch, desc="讀取世代"):
            epoch = future_to_epoch[future]
            try:
                reward_list = future.result()
                if reward_list is not None:
                    all_rewards.append(reward_list)
            except Exception as e:
                print(f"處理世代 {epoch} 時發生錯誤: {e}")
    
    if not all_rewards:
        print("沒有成功載入任何世代的數據")
        return
    
    print("正在處理數據...")
    
    # 合併所有世代的數據
    reward_list = np.concatenate(all_rewards)
    
    # 獎勵名稱列表
    reward_names = [
        "人物偵測",          # Person Detection
        "距離接近",          # Distance+
        "距離遠離",          # Distance-
        "視野內",           # InView
        "視野距離改善",      # ViewDist+
        "視野距離惡化",      # ViewDist-
        "失去視野",         # LostView
        "移動接近",         # Movement+
        "移動遠離",         # Movement-
        "顛倒",            # UpsideDown
        "接觸目標",         # Touch
        "持續接觸"          # Continuous
    ]
    
    # 計算總步數和總和獎勵
    total_steps = len(reward_list)
    steps = np.arange(total_steps)
    total_rewards = np.sum(reward_list, axis=1)
    
    print("正在生成圖表...")
    
    # 創建子圖
    fig = plt.figure(figsize=(15, 24))
    gs = fig.add_gridspec(5, 3)  # 5行3列，最後一行用於總和獎勵
    
    # 設置主標題
    fig.suptitle(f'世代 {start_epoch} 到 {end_epoch} 獎勵分析', fontsize=16, y=0.95)
    
    # 在最上面添加統計信息
    stats_text = (
        f"總步數: {total_steps}\n"
        f"平均總獎勵: {np.mean(total_rewards):.2f}\n"
        f"獎勵標準差: {np.std(total_rewards):.2f}\n"
        f"最大獎勵: {np.max(total_rewards):.2f}\n"
        f"最小獎勵: {np.min(total_rewards):.2f}"
    )
    fig.text(0.02, 0.96, stats_text, fontsize=10, ha='left', va='top')
    
    # 繪製個別獎勵曲線
    axes = []
    for i in range(12):
        ax = fig.add_subplot(gs[i//3, i%3])
        axes.append(ax)
        name = reward_names[i]
        rewards = reward_list[:, i]
        
        # 繪製原始數據的散點圖（透明度較低）
        ax.scatter(steps, rewards, s=1, alpha=0.1, c='blue')
        
        # 計算移動平均
        window_size = 5000
        if len(rewards) >= window_size:
            moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            ax.plot(steps, moving_avg, 'r-', linewidth=2, label=f'{window_size}步移動平均')
        
        ax.set_title(name)
        ax.set_xlabel('訓練步數')
        ax.set_ylabel('獎勵值')
        ax.grid(True)
        
        # 如果有移動平均線，添加圖例
        if len(rewards) >= window_size:
            ax.legend()
        
        # 設置y軸範圍，確保0點在中間
        max_abs_val = max(abs(rewards.min()), abs(rewards.max()))
        ax.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
    
    # 添加總和獎勵和所有獎勵的組合圖
    ax_total = fig.add_subplot(gs[4, :])
    
    # 繪製所有獎勵曲線（使用細線和低透明度）
    colors = plt.cm.tab20(np.linspace(0, 1, len(reward_names)))
    for i, (name, color) in enumerate(zip(reward_names, colors)):
        rewards = reward_list[:, i]
        # 計算移動平均
        if len(rewards) >= window_size:
            moving_avg = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            ax_total.plot(steps, moving_avg, '-', color=color, alpha=0.8, linewidth=1, label=name)
    
    # 繪製總和獎勵（使用粗線）
    ax_total.plot(steps, total_rewards, 'k-', alpha=0.5, linewidth=1, label='總和獎勵')
    
    # 計算總和獎勵的移動平均
    window_size = 5000
    if len(total_rewards) >= window_size:
        total_moving_avg = pd.Series(total_rewards).rolling(window=window_size, center=True).mean()
        ax_total.plot(steps, total_moving_avg, 'r-', linewidth=3, label='總和獎勵移動平均')
    
    ax_total.set_title('總和獎勵與各分量對比')
    ax_total.set_xlabel('訓練步數')
    ax_total.set_ylabel('獎勵值')
    ax_total.grid(True)
    ax_total.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 設置y軸範圍
    max_abs_val = max(abs(total_rewards.min()), abs(total_rewards.max()))
    ax_total.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
    
    # 調整子圖之間的間距
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # 確保輸出目錄存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("儲存圖表...")
    # 保存圖表
    # 加入日期後綴（可選）
    filename = f'rewards_analysis_ep{start_epoch}-{end_epoch}'
    if date_suffix:
        from datetime import datetime
        filename += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    filename += '.png'
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"獎勵分析圖表已保存至: {output_path}")

def analyze_rewards_stats(data_dir, start_epoch, end_epoch, interval=1):
    """分析獎勵統計信息"""
    print(f"正在讀取世代 {start_epoch} 到 {end_epoch} 的數據...")
    all_rewards = []
    
    # 使用線程池並行讀取數據
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 創建讀取任務
        future_to_epoch = {
            executor.submit(load_rewards_from_epoch, data_dir, epoch, interval): epoch
            for epoch in range(start_epoch, end_epoch + 1)
        }
        
        # 使用tqdm顯示進度
        for future in tqdm(future_to_epoch, desc="讀取世代"):
            epoch = future_to_epoch[future]
            try:
                reward_list = future.result()
                if reward_list is not None:
                    all_rewards.append(reward_list)
            except Exception as e:
                print(f"處理世代 {epoch} 時發生錯誤: {e}")
    
    if not all_rewards:
        print("沒有成功載入任何世代的數據")
        return
    
    # 合併所有世代的數據
    reward_list = np.concatenate(all_rewards)
    
    # 計算各種統計數據
    stats = {
        '總步數': len(reward_list),
        '獎勵統計': {}
    }
    
    # 計算總和獎勵
    total_rewards = np.sum(reward_list, axis=1)
    
    # 對每種獎勵類型進行統計
    reward_names = [
        "人物偵測", "距離接近", "距離遠離", "視野內",
        "視野距離改善", "視野距離惡化", "失去視野", 
        "移動接近", "移動遠離", "顛倒", "接觸目標", "持續接觸"
    ]
    
    for i, name in enumerate(reward_names):
        rewards = reward_list[:, i]
        stats['獎勵統計'][name] = {
            '平均值': np.mean(rewards),
            '標準差': np.std(rewards),
            '最大值': np.max(rewards),
            '最小值': np.min(rewards),
            '非零比例': np.mean(rewards != 0) * 100  # 百分比
        }
    
    # 計算總和獎勵統計
    stats['總和獎勵統計'] = {
        '平均值': np.mean(total_rewards),
        '標準差': np.std(total_rewards),
        '最大值': np.max(total_rewards),
        '最小值': np.min(total_rewards),
        '正值比例': np.mean(total_rewards > 0) * 100  # 百分比
    }
    
    # 輸出統計結果
    print(f"\n世代 {start_epoch} 到 {end_epoch} 獎勵統計分析")
    print("=" * 50)
    print(f"總步數: {stats['總步數']}")
    print("\n總和獎勵統計:")
    for key, value in stats['總和獎勵統計'].items():
        if '比例' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.2f}")
    
    print("\n各獎勵類型統計:")
    for name, reward_stats in stats['獎勵統計'].items():
        print(f"\n{name}:")
        for key, value in reward_stats.items():
            if '比例' in key:
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:.6f}")

def estimate_memory_usage(start_epoch, end_epoch, interval):
    """估計記憶體使用量"""
    # 假設每個reward_list元素使用32位元浮點數（4 bytes）
    bytes_per_element = 4
    # 每個步數有12個獎勵值
    elements_per_step = 12
    # 假設每個世代有10000步
    steps_per_epoch = 10000
    
    total_steps = (end_epoch - start_epoch + 1) * steps_per_epoch
    sampled_steps = total_steps // interval
    
    total_bytes = sampled_steps * elements_per_step * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb

def main():
    """主程式"""
    import argparse
    parser = argparse.ArgumentParser(description='獎勵變化分析工具')
    parser.add_argument('--stats', action='store_true',
                      help='是否只顯示統計資訊而不生成圖表')
    parser.add_argument('--dir', type=str, default='E:/train_log0118/train_log',
                      help='數據目錄路徑')
    parser.add_argument('--start-epoch', type=int, required=True,
                      help='起始世代')
    parser.add_argument('--end-epoch', type=int, required=True,
                      help='結束世代')
    parser.add_argument('--interval', type=int, default=1,
                      help='數據取樣間隔，用於減少數據量')
    parser.add_argument('--date-suffix', action='store_true',
                      help='是否在輸出文件名中加入日期時間')
    parser.add_argument('--no-cache', action='store_true',
                      help='不使用緩存加速')
    
    args = parser.parse_args()
    
    # 估計記憶體使用量
    estimated_mb = estimate_memory_usage(args.start_epoch, args.end_epoch, args.interval)
    print(f"估計記憶體使用量: {estimated_mb:.2f} MB")
    
    if estimated_mb > 1000:  # 超過1GB
        user_input = input("預計使用較大記憶體，是否繼續？(y/n) ")
        if user_input.lower() != 'y':
            print("已取消操作")
            return
    
    try:
        if args.stats:
            analyze_rewards_stats(args.dir, args.start_epoch, args.end_epoch, args.interval)
        else:
            plot_rewards(args.dir, args.start_epoch, args.end_epoch, args.interval, 
                        args.date_suffix, not args.no_cache)
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
