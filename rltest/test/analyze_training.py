import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_training_data(file_path='training_results.json'):
    """載入訓練資料"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 將世代資料轉換為 DataFrame
    df = pd.DataFrame(data['epochs'])
    
    # 將時間戳記轉換為 datetime 物件
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 計算累積成功率
    df['cumulative_success_rate'] = df['success'].cumsum() / (df.index + 1) * 100
    
    return df, data['training_start']

def plot_training_analysis(df):
    """繪製訓練分析圖表"""
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 創建一個 2x2 的子圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('訓練結果分析', fontsize=16)
    
    # 1. 步數分布圖
    sns.boxplot(x='success', y='total_steps', data=df, ax=ax1)
    ax1.set_title('成功與失敗的步數分布')
    ax1.set_xlabel('是否成功')
    ax1.set_ylabel('步數')
    
    # 2. 成功率趨勢
    ax2.plot(df.index, df['cumulative_success_rate'], 'b-')
    ax2.set_title('累積成功率趨勢')
    ax2.set_xlabel('世代')
    ax2.set_ylabel('成功率 (%)')
    ax2.grid(True)
    
    # 3. 執行時間趨勢
    ax3.plot(df.index, df['duration_seconds'], 'g-')
    ax3.set_title('世代執行時間趨勢')
    ax3.set_xlabel('世代')
    ax3.set_ylabel('執行時間 (秒)')
    ax3.grid(True)
    
    # 4. FPS 趨勢
    ax4.plot(df.index, df['fps'], 'r-')
    ax4.set_title('FPS 趨勢')
    ax4.set_xlabel('世代')
    ax4.set_ylabel('FPS')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    plt.close()

def print_statistics(df, training_start):
    """輸出統計資訊"""
    print("\n訓練統計資訊:")
    print("-" * 50)
    print(f"訓練開始時間: {training_start}")
    print(f"總世代數: {len(df)}")
    print(f"成功世代數: {df['success'].sum()}")
    print(f"最終成功率: {df['success'].mean()*100:.2f}%")
    print(f"\n步數統計:")
    print(f"  平均步數: {df['total_steps'].mean():.2f}")
    print(f"  成功時的平均步數: {df[df['success']]['success_step'].mean():.2f}")
    print(f"\n時間統計:")
    print(f"  平均執行時間: {df['duration_seconds'].mean():.2f} 秒")
    print(f"  總執行時間: {df['duration_seconds'].sum()/3600:.2f} 小時")
    print(f"\n效能統計:")
    print(f"  平均 FPS: {df['fps'].mean():.2f}")
    print(f"  最低 FPS: {df['fps'].min():.2f}")
    print(f"  最高 FPS: {df['fps'].max():.2f}")

def main():
    try:
        # 載入資料
        df, training_start = load_training_data()
        
        # 繪製分析圖表
        plot_training_analysis(df)
        print("分析圖表已儲存為 'training_analysis.png'")
        
        # 輸出統計資訊
        print_statistics(df, training_start)
        
    except FileNotFoundError:
        print("找不到訓練資料檔案 'training_results.json'")
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()
