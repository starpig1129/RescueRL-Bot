import traceback
import time
import zmq
import sys

try:
    print("開始測試ServerManager（增強版）...")
    
    # 檢查Python和ZeroMQ版本
    print(f"Python版本: {sys.version}")
    print(f"ZeroMQ版本: {zmq.zmq_version()}")
    print(f"PyZMQ版本: {zmq.__version__}")
    
    from server_manager import ServerManager
    
    # 修改連接參數
    custom_ports = {
        'control': 5001,  # 原始端口: 5000
        'info': 8001,     # 原始端口: 8000
        'obs': 6001,      # 原始端口: 6000
        'reset': 7001,    # 原始端口: 7000
        'top_camera': 9001 # 原始端口: 9000
    }
    
    print("導入ServerManager成功，開始初始化...")
    sm = ServerManager()
    
    # 修改服務器地址
    sm.control_address = f'tcp://*:{custom_ports["control"]}'
    sm.info_address = f'tcp://*:{custom_ports["info"]}'
    sm.obs_address = f'tcp://*:{custom_ports["obs"]}'
    sm.reset_address = f'tcp://*:{custom_ports["reset"]}'
    sm.top_camera_address = f'tcp://*:{custom_ports["top_camera"]}'
    
    print(f"使用自定義端口: {custom_ports}")
    print("ServerManager初始化成功")
    
    # 修改超時設置
    sm.heartbeat_interval = 1.0  # 減少心跳間隔（原始值: 5.0）
    sm.heartbeat_timeout = 20.0  # 增加心跳超時（原始值: 10.0）
    sm.connection_timeout = 60.0  # 增加連接超時（原始值: 30.0）
    
    print("開始設置所有服務器...")
    sm.setup_all_servers()
    print("所有服務器設置成功")
    
    # 等待更長時間，確保連接建立
    print("等待連接建立（60秒）...")
    for i in range(12):
        time.sleep(5)
        # 檢查連接狀態
        print(f"[{i*5}秒] 連接狀態:")
        print(f"  控制連接: {sm.control_connected}")
        print(f"  信息連接: {sm.info_connected}")
        print(f"  觀察連接: {sm.obs_connected}")
        print(f"  重置連接: {sm.reset_connected}")
        print(f"  頂部相機連接: {sm.top_camera_connected}")
        print(f"  Unity客戶端連接: {sm.is_unity_connected()}")
        
        # 如果已連接，提前退出等待
        if sm.is_unity_connected():
            print("Unity客戶端已連接！")
            break
    
    # 最終連接狀態
    print("\n最終連接狀態:")
    print(f"控制連接狀態: {sm.control_connected}")
    print(f"信息連接狀態: {sm.info_connected}")
    print(f"觀察連接狀態: {sm.obs_connected}")
    print(f"重置連接狀態: {sm.reset_connected}")
    print(f"頂部相機連接狀態: {sm.top_camera_connected}")
    print(f"Unity客戶端連接狀態: {sm.is_unity_connected()}")
    
    # 如果連接成功，嘗試接收數據
    if sm.is_unity_connected():
        print("\n嘗試接收數據...")
        try:
            image, _ = sm.receive_image(show=False)
            if image is not None:
                print("成功接收圖像數據")
            else:
                print("接收圖像數據失敗")
                
            info = sm.receive_info()
            if info is not None:
                print(f"成功接收信息數據: {info}")
            else:
                print("接收信息數據失敗")
        except Exception as e:
            print(f"接收數據時發生錯誤: {e}")
    
    # 提示用戶修改Unity端的連接地址
    print("\n請在Unity中修改CommunicationManager.cs中的連接參數:")
    print(f"private const string Host = \"127.0.0.1\";")
    print(f"private const int ControlPort = {custom_ports['control']};")
    print(f"private const int InfoPort = {custom_ports['info']};")
    print(f"private const int ObsPort = {custom_ports['obs']};")
    print(f"private const int ResetPort = {custom_ports['reset']};")
    print(f"private const int TopCameraPort = {custom_ports['top_camera']};")
    
    # 等待用戶輸入
    input("\n按Enter鍵關閉服務器...")
    
    # 關閉服務器
    print("關閉服務器...")
    sm.close()
    print("服務器已關閉")
    
except Exception as e:
    print(f"錯誤: {e}")
    traceback.print_exc()