import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import PretrainedResNet, CustomActor, CustomCritic
import gym
from DataReader import DataReader
from CrawlerEnv import CrawlerEnv
import random
from scipy.stats import skew, kurtosis, entropy, gaussian_kde

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def load_sequential_observations(train_log_dir):
    """從訓練數據中獲取連續60張觀察影像，並按照每6幀取1幀的方式採樣10幀"""
    # 初始化 DataReader
    data_reader = DataReader(base_dir=train_log_dir)
    
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
    
    # 隨機選擇起始步數，確保有足夠的後續步數
    random_step = random.randint(0, max_steps-61)  # -61 確保有60張連續影像
    
    # 讀取連續60張影像的數據
    data = data_reader.load_range_data(random_epoch, slice(random_step, random_step+60))
    if data is None or 'origin_image' not in data:
        raise ValueError("無法讀取觀察影像")
    
    # 獲取原始影像並預處理
    images = data['origin_image']  # shape: (60, H, W, C)
    
    # 標準化並轉換為 tensor
    image_tensors = torch.FloatTensor(images / 255.0).permute(0, 3, 1, 2)  # (60, C, H, W)
    
    # 每6幀取1幀，總共取10幀
    sampled_indices = list(range(59, -1, -6))[:10]  # 從最後一幀開始，每隔6幀取一幀
    sampled_images = image_tensors[sampled_indices]
    
    print(f"已隨機選取世代 {random_epoch} 步數 {random_step} 到 {random_step+59} 的連續觀察影像")
    print(f"採樣間隔為6，共選取10幀，採樣索引: {sampled_indices}")
    return sampled_images

def extract_features(model, images):
    """使用模型的特徵提取器處理連續影像"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 將圖像移至相同設備
    images = images.to(device)
    
    # 獲取特徵提取器
    feature_extractor = model.policy.features_extractor
    # 提取特徵
    features_list = []
    with torch.no_grad():
        for i in range(images.shape[0]):
            features = feature_extractor(images[i:i+1])
            features_list.append(features)
    
    # 堆疊所有特徵
    features_tensor = torch.cat(features_list, dim=0)
    return features_tensor.cpu().numpy()

def analyze_network_architecture(model):
    """分析 CustomActor 和 CustomCritic 的網絡架構"""
    # 獲取 Actor 和 Critic 網絡
    actor = model.policy.action_net
    critic = model.policy.value_net
    
    # 分析 Actor 網絡
    print("\nActor 網絡架構分析:")
    print("時序模組 (LSTM):")
    print(f"  輸入維度: {actor.temporal.lstm.input_size}")
    print(f"  隱藏維度: {actor.temporal.lstm.hidden_size}")
    print(f"  LSTM層數: {actor.temporal.lstm.num_layers}")
    print("全連接層:")
    print(f"  FC1: {actor.fc1.in_features} -> {actor.fc1.out_features}")
    print(f"  FC2: {actor.fc2.in_features} -> {actor.fc2.out_features}")
    print(f"Dropout率: {actor.dropout.p}")
    
    # 分析 Critic 網絡
    print("\nCritic 網絡架構分析:")
    print("時序模組 (LSTM):")
    print(f"  輸入維度: {critic.temporal.lstm.input_size}")
    print(f"  隱藏維度: {critic.temporal.lstm.hidden_size}")
    print(f"  LSTM層數: {critic.temporal.lstm.num_layers}")
    print("全連接層:")
    print(f"  FC1: {critic.fc1.in_features} -> {critic.fc1.out_features}")
    print(f"  FC2: {critic.fc2.in_features} -> {critic.fc2.out_features}")
    print(f"Dropout率: {critic.dropout.p}")

def visualize_temporal_features(temporal_data, model_name):
    """視覺化時序特徵的特性"""
    plt.figure(figsize=(15, 5))
    
    # 繪製時序特徵熱圖
    plt.subplot(1, 2, 1)
    plt.imshow(temporal_data[0].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='特徵值')
    plt.title(f'時序特徵熱圖\n{os.path.basename(model_name)}')
    plt.xlabel('時間步')
    plt.ylabel('特徵維度')
    
    # 繪製時序特徵的平均活化值隨時間的變化
    plt.subplot(1, 2, 2)
    mean_activation = np.mean(temporal_data, axis=2)
    for i in range(mean_activation.shape[0]):
        plt.plot(mean_activation[i], label=f'Sample {i}')
    plt.title('時序特徵平均活化值')
    plt.xlabel('時間步')
    plt.ylabel('平均活化值')
    if mean_activation.shape[0] > 1:
        plt.legend()
    
    plt.tight_layout()
    # 確保輸出目錄存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'temporal_features_{os.path.basename(model_name)}.png'))
    plt.close()

def visualize_network_outputs(action_probs, value_pred, model_name):
    """視覺化網絡輸出的特性"""
    plt.figure(figsize=(15, 5))
    
    # 繪製動作概率分布
    plt.subplot(1, 2, 1)
    plt.bar(range(action_probs.shape[-1]), action_probs[0])
    plt.title(f'動作概率分布\n{os.path.basename(model_name)}')
    plt.xlabel('動作索引')
    plt.ylabel('概率')
    
    # 繪製價值預測分布
    plt.subplot(1, 2, 2)
    plt.hist(value_pred.flatten(), bins=30, density=True)
    plt.title('價值預測分布')
    plt.xlabel('預測價值')
    plt.ylabel('密度')
    
    plt.tight_layout()
    # 確保輸出目錄存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'network_outputs_{os.path.basename(model_name)}.png'))
    plt.close()

def analyze_temporal_features(model, features):
    """分析時序特徵處理的特性"""
    # 使用模型的時序特徵處理
    temporal_features = model.policy._get_temporal_features(features)
    
    # 分析時序特徵的統計特性
    print("\n時序特徵分析:")
    print(f"時序特徵形狀: {temporal_features.shape}")
    
    # 計算時序特徵的統計量
    temporal_data = temporal_features.detach().cpu().numpy()
    print("時序特徵統計:")
    print(f"  平均活化值: {np.mean(temporal_data):.6f}")
    print(f"  活化值標準差: {np.std(temporal_data):.6f}")
    print(f"  最大活化值: {np.max(temporal_data):.6f}")
    print(f"  最小活化值: {np.min(temporal_data):.6f}")
    print(f"  時序特徵L2範數: {np.linalg.norm(temporal_data):.6f}")
    
    # 分析時序相關性（添加錯誤處理和數值檢查）
    if temporal_data.shape[1] > 1:  # 確保有多個時間步
        try:
            # 重塑數據以計算相關性
            reshaped_data = temporal_data.reshape(temporal_data.shape[0], -1)
            
            # 檢查數據是否包含無效值
            if np.any(np.isnan(reshaped_data)) or np.any(np.isinf(reshaped_data)):
                print("  警告: 檢測到無效的特徵值，跳過相關性計算")
                return temporal_features
            
            # 檢查數據是否有變化
            if np.all(reshaped_data == reshaped_data[0]):
                print("  警告: 特徵值無變化，無法計算相關性")
                return temporal_features
            
            # 檢查數據的變異性
            data_std = np.std(reshaped_data)
            if data_std < 1e-6:
                print("  警告: 數據變異性過小，特徵幾乎沒有變化")
                print(f"  數據標準差: {data_std:.6f}")
                return temporal_features

            # 標準化數據以避免數值計算問題
            normalized_data = (reshaped_data - np.mean(reshaped_data, axis=0)) / (np.std(reshaped_data, axis=0) + 1e-10)
            
            # 計算相關性
            temporal_corr = np.corrcoef(normalized_data.T)
            
            # 檢查相關係數矩陣是否有效
            if np.any(np.isnan(temporal_corr)) or np.any(np.isinf(temporal_corr)):
                print("  警告: 相關係數計算出現無效值")
                print("  可能原因: 數據中存在常數列或標準差為零的列")
                return temporal_features
            
            # 移除對角線元素（自相關）
            mask = ~np.eye(temporal_corr.shape[0], dtype=bool)
            mean_corr = np.mean(np.abs(temporal_corr[mask]))
            
            print(f"  時間步間平均相關性: {mean_corr:.6f}")
            print(f"  數據變異程度: {data_std:.6f}")
            
            # 檢查相關性分布
            corr_std = np.std(temporal_corr[mask])
            print(f"  相關係數標準差: {corr_std:.6f}")
            if corr_std < 1e-6:
                print("  警告: 相關係數幾乎完全相同，可能表示特徵提取不夠有效")
        except Exception as e:
            print(f"  警告: 計算相關性時發生錯誤: {str(e)}")
    
    return temporal_features

def analyze_network_outputs(model, temporal_features, model_name):
    """分析網絡輸出的特性"""
    with torch.no_grad():
        # 獲取 Actor 輸出（動作分布）
        action_logits = model.policy.action_net(temporal_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # 獲取 Critic 輸出（價值預測）
        value_pred = model.policy.value_net(temporal_features)
        
        # 轉換為numpy進行分析
        action_probs = action_probs.cpu().numpy()
        value_pred = value_pred.cpu().numpy()
        
        print("\n網絡輸出分析:")
        print("Actor輸出 (動作概率):")
        print(f"  動作數量: {action_probs.shape[-1]}")
        print(f"  最高概率: {np.max(action_probs):.6f}")
        print(f"  最低概率: {np.min(action_probs):.6f}")
        print(f"  平均概率: {np.mean(action_probs):.6f}")
        print(f"  概率熵: {-np.sum(action_probs * np.log(action_probs + 1e-10)):.6f}")
        
        print("\nCritic輸出 (價值預測):")
        print(f"  平均預測值: {np.mean(value_pred):.6f}")
        print(f"  預測值標準差: {np.std(value_pred):.6f}")
        print(f"  最大預測值: {np.max(value_pred):.6f}")
        print(f"  最小預測值: {np.min(value_pred):.6f}")
        
        # 視覺化網絡輸出
        visualize_network_outputs(action_probs, value_pred, model_name)

def analyze_network_parameters(model):
    """分析網絡參數的統計特性"""
    actor = model.policy.action_net
    critic = model.policy.value_net
    
    def get_param_stats(named_parameters, name):
        stats_dict = {}
        for param_name, param in named_parameters:
            if param.requires_grad:
                data = param.detach().cpu().numpy()
                stats_dict[f"{name}_{param_name}"] = {
                    '平均值': np.mean(data),
                    '標準差': np.std(data),
                    'L2範數': np.linalg.norm(data),
                    '最大值': np.max(data),
                    '最小值': np.min(data),
                    '參數量': data.size
                }
        return stats_dict
    
    # 獲取 Actor 和 Critic 的參數統計
    actor_stats = get_param_stats(actor.named_parameters(), 'actor')
    critic_stats = get_param_stats(critic.named_parameters(), 'critic')
    
    # 打印統計信息
    print("\n網絡參數統計分析:")
    for name, stats in {**actor_stats, **critic_stats}.items():
        print(f"\n{name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.6f}" if isinstance(value, float) else f"  {stat_name}: {value}")

def analyze_feature_statistics(features):
    """分析特徵的統計特性"""
    stats_dict = {
        '平均值': np.mean(features),
        '標準差': np.std(features),
        '中位數': np.median(features),
        '最小值': np.min(features),
        '最大值': np.max(features),
        '偏度': skew(features.flatten()),
        '峰度': kurtosis(features.flatten()),
        '非零元素比例': np.mean(features != 0),
        'L1範數': np.linalg.norm(features, ord=1),
        'L2範數': np.linalg.norm(features, ord=2)
    }
    return stats_dict

def compare_features(model_paths,train_log_dir):
    """比較不同模型的特徵提取結果"""
    # 載入連續的觀察影像
    images = load_sequential_observations(train_log_dir)
    
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
            
            # 分析網絡架構和參數
            print(f"\n分析模型: {os.path.basename(model_path)}")
            analyze_network_architecture(model)
            analyze_network_parameters(model)
            
            # 提取特徵並進行分析
            features = extract_features(model, images)  # (sequence_length, feature_dim)
            
            # 初始化模型的特徵緩衝區
            model.policy.feature_buffer_tensor = torch.zeros(
                (1, model.policy.buffer_size, features.shape[-1]),
                dtype=torch.float32,
                device=device
            )
            
            # 逐步處理每個時間步的特徵
            features_tensor = torch.from_numpy(features).to(device)
            for i in range(features.shape[0]):
                # 將每個時間步的特徵添加到緩衝區
                model.policy.feature_buffer_tensor = torch.cat([
                    model.policy.feature_buffer_tensor[:, 1:],
                    features_tensor[i:i+1].unsqueeze(0)
                ], dim=1)
            
            # 分析時序特徵
            temporal_features = analyze_temporal_features(model, features_tensor)
            temporal_data = temporal_features.detach().cpu().numpy()
            visualize_temporal_features(temporal_data, model_path)
            
            # 分析網絡輸出
            analyze_network_outputs(model, temporal_features, model_path)
            all_features.append(features)
            model_names.append(os.path.basename(model_path))
        except FileNotFoundError:
            print(f"找不到模型文件: {model_path}")
            continue
    
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
            kl_div = entropy(np.histogram(all_features[i], bins=50)[0] + 1e-10,
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
        density = gaussian_kde(features.flatten())
        xs = np.linspace(features.min(), features.max(), 200)
        plt.plot(xs, density(xs), 'r-', lw=2)
        plt.title(f'特徵分布\n{name}')
        plt.xlabel('特徵值')
        plt.ylabel('密度')
    
    plt.tight_layout()
    # 確保輸出目錄存在
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'feature_comparison.png'))
    plt.close()
    
    # 計算並視覺化特徵相關性
    if len(all_features) < 2:
        print("警告: 需要至少兩個模型的特徵才能計算相關性")
        return

    # 檢查特徵數據是否有效
    for features in all_features:
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("警告: 特徵中包含無效值(NaN/Inf)")
            return
        
    try:
        # 確保特徵向量被正確展平且標準化
        flattened_features = [f.flatten() for f in all_features]
        # 添加小的常數避免除以零
        correlation_matrix = np.corrcoef(flattened_features)
        
        # 檢查相關性矩陣是否有效
        if correlation_matrix.size == 0 or np.any(np.isnan(correlation_matrix)):
            print("警告: 無法計算有效的相關性矩陣")
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.yticks(range(len(model_names)), model_names)
        plt.title('模型特徵相關性矩陣')
        plt.tight_layout()
        
        # 確保輸出目錄存在
        output_dir = "analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
        plt.close()
        
    except Exception as e:
        print(f"計算相關性矩陣時發生錯誤: {str(e)}")

def main():
    try:
        # 設定模型路徑
        model_paths = [
            "E:/train_log0118/models/ppo_crawler_ep050.zip",   # 訓練前的模型
            "E:/train_log0118/models/ppo_crawler_ep2000.zip",  # 訓練中期的模型
            "E:/train_log0118/models/ppo_crawler_ep5650.zip"  # 訓練後期的模型
        ]
        
        # 檢查模型文件是否存在
        for path in model_paths:
            if not os.path.exists(path):
                print(f"警告: 模型文件不存在: {path}")
        
        # 檢查訓練日誌目錄
        train_log_dir = "E:/train_log0118/train_log"
        if not os.path.exists(train_log_dir):
            raise FileNotFoundError(f"找不到訓練日誌目錄: {train_log_dir}")
        
        # 執行特徵比較
        compare_features(model_paths,train_log_dir)
        print(f"\n分析結果已保存至 'analysis_results' 目錄")
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
