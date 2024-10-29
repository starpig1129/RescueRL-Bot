# test.py
import os
import signal
import sys
from stable_baselines3 import PPO
from CrawlerEnv import CrawlerEnv
from policy import CustomPolicy, PretrainedResNet
import torch
import time
from DataHandler import DataHandler
# 全局變量
env = None
model = None
# 設置模型參數
model_params = {
    "policy": CustomPolicy,
    "env": None,  # 暫時設為 None，稍後會設置為 env
    "verbose": 2,
    "learning_rate": 3e-4, # Increased learning rate
    "n_steps": 2048,
    "batch_size": 128, # Increased batch size
    "n_epochs": 20, # Increased number of epochs
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": 4,
    "target_kl": 0.03,
    "tensorboard_log": "./logs/",
}
def signal_handler(sig, frame):
    """處理 Ctrl+C 信號，確保環境關閉"""
    print('\n收到中斷信號 (Ctrl+C)，正在關閉...')
    if env is not None:
        env.close()
    sys.exit(0)
def format_time(seconds):
    """格式化時間為 HH:MM:SS 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
# 設置信號處理器來處理 Ctrl+C 中斷
signal.signal(signal.SIGINT, signal_handler)

def get_all_models(model_dir="models"):
    """獲取所有模型檔案，按照世代排序"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目錄 {model_dir} 不存在")
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith("ppo_crawler_ep") and f.endswith(".zip")]
    if not model_files:
        raise FileNotFoundError("找不到任何模型檔案")

    # 提取檔案中的世代號碼並排序
    models = [(int(f.split('_ep')[1].split('.')[0]), f) for f in model_files]
    models.sort(key=lambda x: x[0])  # 按世代排序
    
    return [(os.path.join(model_dir, f), epoch) for epoch, f in models]

def run_single_test(model_path, epoch, env, episodes, data_handler, render=True):
    """运行单个模型的测试"""
    print(f"\n开始测试模型: {model_path} (世代 {epoch})")
    
    # 加载模型
    custom_objects = {
        'policy_class': CustomPolicy,
    }
    
    try:
        model = PPO.load(
            model_path,
            env=env,
            custom_objects={'policy_class': CustomPolicy},
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None
    
    # 运行测试集数
    for episode in range(episodes):
        print(f"\n开始测试集数 {episode + 1}/{episodes}")
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # 获取中间输出
            features_extractor = model.policy.features_extractor
            layer_outputs = features_extractor.layer_outputs
            action_logits = model.policy.action_logits
            
            # 准备数据字典
            data = {
                'step': step,
                'obs': obs,
                'reward': reward,
                'layer_outputs': {
                    'layer_input': layer_outputs.get('input'),
                    'layer_conv1': layer_outputs.get('conv1_output'),
                    'layer_final_residual': layer_outputs.get('layer4_output'),
                    'layer_feature': layer_outputs.get('features_output'),
                    'layer_actor': action_logits
                }
            }
            
            # 保存数据
            data_handler.save_step_data(**data)
            
            if step % 100 == 0:
                print(f"步数: {step}, 当前累计奖励: {total_reward:.2f}")
        
        total_steps_all_episodes += step
        total_reward_all_episodes += total_reward
        
        print(f"测试集数 {episode + 1} 完成")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward:.2f}")
    
    # 计算平均绩效
    avg_steps = total_steps_all_episodes / episodes
    avg_reward = total_reward_all_episodes / episodes
    
    return {
        'epoch': epoch,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'total_steps': total_steps_all_episodes,
        'total_reward': total_reward_all_episodes
    }


def run_test(episodes=5, model_path=None, render=True):
    """
    运行测试
    
    Args:
        episodes (int): 要运行的测试集数
        model_path (str): 指定的模型路径，如果为 None 则测试所有模型
        render (bool): 是否显示可视化界面
    """
    global env, data_handler
    
    try:
        # 准备测试的模型列表
        if model_path is not None:
            # 单一模型测试
            epoch = int(model_path.split('_ep')[-1].split('.')[0])
            models_to_test = [(model_path, epoch)]
        else:
            # 测试所有模型
            models_to_test = get_all_models()
        
        # 初始化测试结果列表
        test_results = []
        start_time = time.time()
        
        # 初始化 DataHandler
        data_handler = DataHandler(base_dir="test_logs")
        
        # 为每个模型创建环境并运行测试
        for model_path, epoch in models_to_test:
            # 为每个模型创建新的环境
            if env is not None:
                env.close()
            env = CrawlerEnv(show=render, epoch=epoch, test_mode=True)
            model_params['env'] = env  # 將環境傳遞給模型
            # 创建新的 epoch 文件
            data_handler.create_epoch_file(epoch)
            
            # 运行测试并收集结果
            result = run_single_test(model_path, epoch, env, episodes, data_handler, render)
            if result is not None:
                test_results.append(result)
            
            # 关闭当前 epoch 文件
            data_handler.close_epoch_file()
            
            # 显示当前进度
            elapsed_time = time.time() - start_time
            if len(models_to_test) > 1:
                progress = (len(test_results) / len(models_to_test)) * 100
                print(f"\n测试进度: {progress:.1f}% ({len(test_results)}/{len(models_to_test)})")
                print(f"已用时间: {format_time(elapsed_time)}")
        
        # 显示总结果
        if test_results:
            print("\n========= 测试结果总结 =========")
            print(f"测试完成的模型数量: {len(test_results)}")
            print(f"每个模型的测试集数: {episodes}")
            print(f"总用时: {format_time(time.time() - start_time)}")
            print("\n各模型平均绩效:")
            print("世代\t平均步数\t平均奖励")
            print("-" * 40)
            for result in test_results:
                print(f"{result['epoch']}\t{result['avg_steps']:.1f}\t\t{result['avg_reward']:.2f}")
            
            # 找出最佳模型
            best_model = max(test_results, key=lambda x: x['avg_reward'])
            print("\n最佳模型:")
            print(f"世代: {best_model['epoch']}")
            print(f"平均步数: {best_model['avg_steps']:.1f}")
            print(f"平均奖励: {best_model['avg_reward']:.2f}")
        else:
            print("\n没有成功完成测试的模型")
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        if env is not None:
            print("\n关闭环境...")
            env.close()


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crawler 模型測試程式')
    parser.add_argument('--episodes', type=int, default=5,
                      help='每個模型要運行的測試集數 (預設: 5)')
    parser.add_argument('--model', type=str, default=None,
                      help='指定的模型路徑 (預設: 測試所有模型)')
    parser.add_argument('--render', type=bool, default=False,
                      help='不顯示視覺化界面')
    
    args = parser.parse_args()
    
    try:
        run_test(
            episodes=args.episodes,
            model_path=args.model,
            render=args.render
        )
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
    except Exception as e:
        print(f"執行測試時發生錯誤: {e}")

if __name__ == "__main__":
    main()