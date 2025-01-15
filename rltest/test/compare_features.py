import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
from policy import PretrainedResNet
import gym
from DataReader import DataReader
from CrawlerEnv import CrawlerEnv
import random
from scipy import stats

def load_random_observation():
    """從訓練數據中隨機獲取一張觀察影像"""
    # 初始化 DataReader
    data_reader = DataReader(base_dir="train_logs")
    
    # 獲取所有可用的世代
    epochs = data_reader.get_all_epochs()
    if not epochs:
        raise ValueError("沒有找到可用的訓練數據")
    
    # 隨機選擇一個世代
    random_epoch = random.choice(epochs)
    
    # 獲取該世代的最大步數
    max_steps = data_reader.get_max_steps(random_epoch)
    if not max_steps:
        raise ValueError(f"無法獲取世代 {random_epoch} 的步數")
    
    # 隨機選擇一個步數
    random_step = random.randint(0, max_steps-1)
    
    # 讀取數據
    data = data_reader.load_range_data(random_epoch, slice(random_step, random_step+1))
    if data is None or 'origin_image' not in data:
        raise ValueError("無法讀取觀察影像")
    
    # 獲取原始影像並預處理
    image = data['origin_image'][0]  # shape: (H, W, C)
    
    # 標準化並轉換為 tensor
    image_tensor = torch.FloatTensor(image / 255.0).permute(2, 0, 1).unsqueeze(0)
    
    print(f"已隨機選取世代 {random_epoch} 步數 {random_step} 的觀察影像")
    return image_tensor

def extract_features(model, image):
    """使用模型的特徵提取器處理影像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 將圖像移至相同設備
    image = image.to(device)
    
    # 獲取特徵提取器
    feature_extractor = model.policy.features_extractor
    # 提取特徵
    with torch.no_grad():
        features = feature_extractor(image)
    return features.cpu().numpy()

def analyze_feature_statistics(features):
    """分析特徵的統計特性"""
    stats_dict = {
        '平均值': np.mean(features),
        '標準差': np.std(features),
        '中位數': np.median(features),
        '最小值': np.min(features),
        '最大值': np.max(features),
        '偏度': stats.skew(features.flatten()),
        '峰度': stats.kurtosis(features.flatten()),
        '非零元素比例': np.mean(features != 0),
        'L1範數': np.linalg.norm(features, ord=1),
        'L2範數': np.linalg.norm(features, ord=2)
    }
    return stats_dict

def compare_features(model_paths):
    """比較不同模型的特徵提取結果"""
    # 載入隨機觀察影像
    image = load_random_observation()
    
    # 儲存每個模型的特徵
    all_features = []
    model_names = []
    
    # 檢測是否有 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入每個模型並提取特徵
    for model_path in model_paths:
        try:
            model = PPO.load(model_path, device=device)
        except FileNotFoundError:
            print(f"找不到模型文件: {model_path}")
            continue
        features = extract_features(model, image)
        all_features.append(features)
        model_names.append(os.path.basename(model_path))
    
    # 分析每個模型的特徵統計特性
    print("\n特徵統計分析:")
    for i, (features, name) in enumerate(zip(all_features, model_names)):
        print(f"\n模型 {name} 的特徵統計:")
        stats = analyze_feature_statistics(features)
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value:.6f}")
    
    # 計算特徵差異
    print("\n特徵差異分析:")
    for i in range(len(all_features)):
        for j in range(i + 1, len(all_features)):
            # 計算多種差異指標
            abs_diff = np.mean(np.abs(all_features[i] - all_features[j]))
            cosine_sim = np.dot(all_features[i].flatten(), all_features[j].flatten()) / \
                        (np.linalg.norm(all_features[i]) * np.linalg.norm(all_features[j]))
            kl_div = stats.entropy(np.histogram(all_features[i], bins=50)[0] + 1e-10,
                                 np.histogram(all_features[j], bins=50)[0] + 1e-10)
            
            print(f"\n{model_names[i]} vs {model_names[j]}:")
            print(f"  平均絕對差異: {abs_diff:.6f}")
            print(f"  餘弦相似度: {cosine_sim:.6f}")
            print(f"  KL散度: {kl_div:.6f}")
    
    # 視覺化特徵
    plt.figure(figsize=(15, 5))
    
    # 繪製每個模型的特徵分布
    for i, (features, name) in enumerate(zip(all_features, model_names)):
        plt.subplot(1, len(all_features), i + 1)
        plt.hist(features.flatten(), bins=50, alpha=0.7, density=True)
        # 添加核密度估計
        density = stats.gaussian_kde(features.flatten())
        xs = np.linspace(features.min(), features.max(), 200)
        plt.plot(xs, density(xs), 'r-', lw=2)
        plt.title(f'特徵分布\n{name}')
        plt.xlabel('特徵值')
        plt.ylabel('密度')
    
    plt.tight_layout()
    plt.savefig('feature_comparison.png')
    plt.close()
    
    # 計算並視覺化特徵相關性
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef([f.flatten() for f in all_features])
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(model_names)), model_names, rotation=45)
    plt.yticks(range(len(model_names)), model_names)
    plt.title('模型特徵相關性矩陣')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()

def main():
    try:
        # 設定模型路徑
        model_paths = [
            "models/ppo_crawler_ep100.zip",   # 訓練前的模型
            "models/ppo_crawler_ep300.zip",  # 訓練中期的模型
            "models/ppo_crawler_ep400.zip"  # 訓練後期的模型
        ]
        
        # 檢查模型文件是否存在
        for path in model_paths:
            if not os.path.exists(path):
                print(f"警告: 模型文件不存在: {path}")
        
        # 檢查訓練日誌目錄
        if not os.path.exists("train_logs"):
            raise FileNotFoundError("找不到 train_logs 目錄")
        
        # 執行特徵比較
        compare_features(model_paths)
        print("\n分析結果已保存為 'feature_comparison.png' 和 'feature_correlation.png'")
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
