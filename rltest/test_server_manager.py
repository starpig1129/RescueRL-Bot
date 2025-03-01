import traceback
import time

try:
    print("開始測試ServerManager...")
    from server_manager import ServerManager
    
    print("導入ServerManager成功，開始初始化...")
    sm = ServerManager()
    print("ServerManager初始化成功")
    
    print("開始設置所有服務器...")
    sm.setup_all_servers()
    print("所有服務器設置成功")
    
    # 等待一段時間，確保連接建立
    print("等待連接建立...")
    time.sleep(5)
    
    # 檢查連接狀態
    print(f"控制連接狀態: {sm.control_connected}")
    print(f"信息連接狀態: {sm.info_connected}")
    print(f"觀察連接狀態: {sm.obs_connected}")
    print(f"重置連接狀態: {sm.reset_connected}")
    print(f"頂部相機連接狀態: {sm.top_camera_connected}")
    
    # 檢查Unity客戶端是否已連接
    print(f"Unity客戶端連接狀態: {sm.is_unity_connected()}")
    
    # 關閉服務器
    print("關閉服務器...")
    sm.close()
    print("服務器已關閉")
    
except Exception as e:
    print(f"錯誤: {e}")
    traceback.print_exc()