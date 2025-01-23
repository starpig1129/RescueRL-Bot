import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_training_data(file_path='training_results.csv', steps_threshold=None):
    """載入訓練資料
    
    Args:
        file_path: CSV檔案路徑
        steps_threshold: 步數閾值，用於過濾實際上不是成功的結果
    """
    # 讀取 CSV 檔案
    df = pd.read_csv(file_path)
    
    # 如果設定了步數閾值，將超過閾值的「成功」案例標記為失敗
    if steps_threshold is not None:
        df.loc[(df['是否成功'] == 1) & ((df['總步數'] - df['成功步數']) > steps_threshold), '是否成功'] = 0
    
    # 計算累積成功率
    df['cumulative_success_rate'] = df['是否成功'].cumsum() / (df.index + 1) * 100
    
    return df

def plot_training_analysis(df, steps_threshold=None):
    """繪製訓練分析圖表"""
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建一個 2x2 的子圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    title = '訓練結果分析'
    if steps_threshold is not None:
        title += f' (步數閾值: {steps_threshold})'
    fig.suptitle(title, fontsize=16)
    
    # 1. 步數分布圖
    sns.boxplot(x='是否成功', y='總步數', data=df, ax=ax1)
    ax1.set_title('成功與失敗的步數分布')
    ax1.set_xlabel('是否成功')
    ax1.set_ylabel('步數')
    
    # 2. 成功率趨勢
    window_size = 50  # 移動平均窗口大小
    rolling_success_rate = df['是否成功'].rolling(window=window_size).mean() * 100
    ax2.plot(df.index, df['cumulative_success_rate'], 'b-', label='累積成功率')
    ax2.plot(df.index, rolling_success_rate, 'r-', label=f'{window_size}世代移動平均')
    ax2.set_title('成功率趨勢')
    ax2.set_xlabel('世代')
    ax2.set_ylabel('成功率 (%)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 成功步數趨勢
    success_df = df[df['是否成功'] == 1]
    if not success_df.empty:
        ax3.plot(success_df['世代'], success_df['成功步數'], 'g-')
        ax3.set_title('成功時的步數趨勢')
        ax3.set_xlabel('世代')
        ax3.set_ylabel('成功步數')
        ax3.grid(True)
        
        # 添加移動平均線
        if len(success_df) >= window_size:
            rolling_steps = success_df['成功步數'].rolling(window=window_size).mean()
            ax3.plot(success_df['世代'], rolling_steps, 'r-', 
                    label=f'{window_size}世代移動平均')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, '尚無成功資料', ha='center', va='center')
    
    # 4. 最小距離與世代關係
    ax4.plot(df.index, df['最小距離'], 'b-')
    if len(df) >= window_size:
        rolling_distance = df['最小距離'].rolling(window=window_size).mean()
        ax4.plot(df.index, rolling_distance, 'r-', label=f'{window_size}世代移動平均')
    ax4.set_title('最小距離趨勢')
    ax4.set_xlabel('世代')
    ax4.set_ylabel('最小距離')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    plt.close()

def print_statistics(df):
    """輸出統計資訊"""
    print("\n訓練統計資訊:")
    print("-" * 50)
    print(f"總世代數: {len(df)}")
    print(f"成功世代數: {df['是否成功'].sum()}")
    print(f"最終成功率: {df['是否成功'].mean()*100:.2f}%")
    
    # 計算最近100世代的成功率
    if len(df) >= 100:
        recent_success_rate = df['是否成功'].tail(100).mean() * 100
        print(f"最近100世代成功率: {recent_success_rate:.2f}%")
    
    print(f"\n步數統計:")
    print(f"  平均總步數: {df['總步數'].mean():.2f}")
    success_steps = df[df['是否成功'] == 1]['成功步數']
    if not success_steps.empty:
        print(f"  成功時的平均步數: {success_steps.mean():.2f}")
        print(f"  最快成功步數: {success_steps.min():.0f}")
        print(f"  最慢成功步數: {success_steps.max():.0f}")
        
        if len(success_steps) >= 10:
            recent_steps = success_steps.tail(10).mean()
            print(f"  最近10次成功的平均步數: {recent_steps:.2f}")
    
    print(f"\n時間統計:")
    print(f"  平均執行時間: {df['執行時間'].mean():.2f} 秒")
    print(f"  總執行時間: {df['執行時間'].sum()/3600:.2f} 小時")
    print(f"  最短執行時間: {df['執行時間'].min():.2f} 秒")
    print(f"  最長執行時間: {df['執行時間'].max():.2f} 秒")

def main():
    try:
        # 設定步數閾值（可以根據需要調整）
        steps_threshold = 100  # 總步數與成功步數的差異閾值
        
        # 載入資料
        df = load_training_data('E:/train_log0118/training_results.csv', steps_threshold)
        
        # 繪製分析圖表
        plot_training_analysis(df, steps_threshold)
        print("分析圖表已儲存為 'training_analysis.png'")
        
        # 輸出統計資訊
        print_statistics(df)
        
    except FileNotFoundError:
        print("找不到訓練資料檔案 'training_results.csv'")
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
