using UnityEngine;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using System.IO;
using System.Text;
using System.Collections;

public class CommunicationManager : MonoBehaviour
{
    // 單例模式
    private static CommunicationManager _instance;
    private static readonly object instanceLock = new object();
    private static bool isInitializing = false;

    public static CommunicationManager Instance
    {
        get
        {
            if (_instance == null)
            {
                // 只在遊戲運行時查找或創建實例
                if (Application.isPlaying)
                {
                    // 先查找場景中是否已存在實例
                    _instance = FindObjectOfType<CommunicationManager>();

                    // 如果不存在，則創建一個新的
                    if (_instance == null)
                    {
                        GameObject go = new GameObject("CommunicationManager");
                        _instance = go.AddComponent<CommunicationManager>();
                        DontDestroyOnLoad(go);
                        Debug.Log("創建新的CommunicationManager實例");
                    }
                }
            }
            return _instance;
        }
    }

    // 各種 ZeroMQ 套接字
    private ZmqSocket controlSocket; // 端口5000
    private ZmqSocket infoSocket;    // 端口8000
    private ZmqSocket obsSocket;     // 端口6000
    private ZmqSocket resetSocket;   // 端口7000
    private ZmqSocket topCameraSocket; // 端口9000

    // 對應的套接字類型
    private ZmqSocketType controlSocketType = ZmqSocketType.Request;   // REQ (Unity) -> REP (Python)
    private ZmqSocketType infoSocketType = ZmqSocketType.Request;      // REQ (Unity) -> REP (Python)
    private ZmqSocketType obsSocketType = ZmqSocketType.Request;       // REQ (Unity) -> REP (Python)
    private ZmqSocketType resetSocketType = ZmqSocketType.Request;     // REQ (Unity) -> REP (Python)
    private ZmqSocketType topCameraSocketType = ZmqSocketType.Request; // REQ (Unity) -> REP (Python)

    // 連接狀態
    private bool isControlConnected;
    private bool isInfoConnected;
    private bool isObsConnected;
    private bool isResetConnected;
    private bool isTopCameraConnected;

    // 連接鎖
    private readonly object controlLock = new object();
    private readonly object infoLock = new object();
    private readonly object obsLock = new object();
    private readonly object resetLock = new object();
    private readonly object topCameraLock = new object();

    // 連接參數
    private const string Host = "127.0.0.1";
    private const int ControlPort = 5001;
    private const int InfoPort = 8001;
    private const int ObsPort = 6001;
    private const int ResetPort = 7001;
    private const int TopCameraPort = 9001;

    // 重連參數
    private const int MaxRetryAttempts = 10;
    private const int RetryDelayMs = 2000;

    // 公開屬性
    public bool IsControlConnected => isControlConnected;
    public bool IsInfoConnected => isInfoConnected;
    public bool IsObsConnected => isObsConnected;
    public bool IsResetConnected => isResetConnected;
    public bool IsTopCameraConnected => isTopCameraConnected;

    // 為了保持與原有代碼的兼容性，我們提供相同的屬性名稱
    // 但實際上返回的是 ZmqSocket 對象
    public ZmqSocket ControlStream => controlSocket;
    public ZmqSocket InfoStream => infoSocket;
    public ZmqSocket ObsStream => obsSocket;
    public ZmqSocket ResetStream => resetSocket;
    public ZmqSocket TopCameraStream => topCameraSocket;

    public object ControlLock => controlLock;
    public object InfoLock => infoLock;
    public object ObsLock => obsLock;
    public object ResetLock => resetLock;
    public object TopCameraLock => topCameraLock;

    // 確保實例存在的靜態方法
    public static CommunicationManager EnsureInstance()
    {
        if (_instance != null)
            return _instance;

        lock (instanceLock)
        {
            if (_instance != null)
                return _instance;

            if (isInitializing)
            {
                Debug.LogWarning("正在初始化CommunicationManager實例，請稍候...");
                // 等待初始化完成
                int waitCount = 0;
                while (isInitializing && waitCount < 100)
                {
                    System.Threading.Thread.Sleep(10);
                    waitCount++;
                }

                if (_instance != null)
                    return _instance;
            }

            isInitializing = true;

            Debug.Log("創建新的CommunicationManager實例");
            GameObject go = new GameObject("CommunicationManager");
            CommunicationManager manager = go.AddComponent<CommunicationManager>();
            DontDestroyOnLoad(go);

            // 確保在返回之前初始化完成
            isInitializing = false;

            return manager;
        }
    }

    // 初始化
    private void Awake()
    {
        // 確保 NetMQ 包裝器已初始化
        NetMQWrapper.Instance.StartNetMQ();

        lock (instanceLock)
        {
            if (_instance == null)
            {
                _instance = this;
                DontDestroyOnLoad(gameObject);
                Debug.Log("CommunicationManager實例已初始化");
            }
            else if (_instance != this)
            {
                Debug.LogWarning("檢測到多個CommunicationManager實例，銷毀重複實例");
                Destroy(gameObject);
            }
        }
    }

    // 確保實例不會被銷毀
    private void OnDestroy()
    {
        if (_instance == this)
        {
            Debug.LogWarning("CommunicationManager實例被銷毀，將在下次需要時重新創建");
            _instance = null;
        }
    }

    // 連接監控相關
    private bool isMonitoringConnections = false;
    private float connectionCheckInterval = 5f; // 每5秒檢查一次
    private float heartbeatInterval = 5f; // 每5秒發送一次心跳
    private float connectionTestTimeout = 10f; // 連接測試超時時間（增加到10秒）

    private void Start()
    {
        // 初始化所有連接
        InitializeAllConnections().ContinueWith(task => {
            if (task.Exception != null)
            {
                Debug.LogError($"初始化連接時發生錯誤: {task.Exception.Message}");
            }
            else
            {
                // 啟動連接監控
                StartConnectionMonitoring();

                // 啟動心跳發送
                StartHeartbeat();
            }
        });
    }

    private void StartConnectionMonitoring()
    {
        if (!isMonitoringConnections)
        {
            isMonitoringConnections = true;
            StartCoroutine(MonitorConnections());
            Debug.Log("已啟動連接監控");
        }
    }

    private void StartHeartbeat()
    {
        // 啟動心跳發送協程
        StartCoroutine(SendHeartbeats());
        Debug.Log("已啟動心跳發送");
    }

    private System.Collections.IEnumerator MonitorConnections()
    {
        while (isMonitoringConnections)
        {
            yield return new WaitForSeconds(connectionCheckInterval);

            // 檢查所有連接
            bool needReconnect = false;

            if (!isControlConnected || controlSocket == null) needReconnect = true;
            if (!isInfoConnected || infoSocket == null) needReconnect = true;
            if (!isObsConnected || obsSocket == null) needReconnect = true;
            if (!isResetConnected || resetSocket == null) needReconnect = true;
            if (!isTopCameraConnected || topCameraSocket == null) needReconnect = true;

            if (needReconnect)
            {
                Debug.LogWarning("檢測到連接斷開，嘗試重新連接...");
                InitializeAllConnections().ContinueWith(task => {
                    if (task.Exception != null)
                    {
                        Debug.LogError($"重新連接時發生錯誤: {task.Exception.Message}");
                    }
                    else
                    {
                        Debug.Log("重新連接成功");
                    }
                });
            }
        }
    }

    private System.Collections.IEnumerator SendHeartbeats()
    {
        while (true)
        {
            // 等待指定的心跳間隔
            yield return new WaitForSeconds(heartbeatInterval);

            // 發送信息通道心跳
            SendInfoHeartbeat();

            // 發送觀察通道心跳
            SendObsHeartbeat();

            // 發送頂部相機通道心跳
            SendTopCameraHeartbeat();

            // 控制通道和重置通道的心跳在實際發送數據時處理
        }
    }

    private async void SendInfoHeartbeat()
    {
        if (!isInfoConnected || infoSocket == null || infoSocketType != ZmqSocketType.Request) return;

        try
        {
            // 創建心跳數據
            long timestamp = (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds;
            string heartbeatJson = "{\"heartbeat\":true,\"timestamp\":" + timestamp + "}";
            byte[] heartbeatData = Encoding.UTF8.GetBytes(heartbeatJson);

            // 發送心跳
            await SendDataAsync(heartbeatData, infoSocket, infoLock, isInfoConnected);

            // 接收回應 (REQ/REP模式需要)
            byte[] response = await ReceiveDataAsync(infoSocket, infoLock, isInfoConnected);
            if (response == null)
            {
                Debug.LogWarning("未收到信息心跳回應");
                isInfoConnected = false;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"發送信息心跳失敗: {e.Message}");
        }
    }

    private async void SendObsHeartbeat()
    {
        if (!isObsConnected || obsSocket == null || obsSocketType != ZmqSocketType.Request) return;

        try
        {
            // 創建一個1x1的空白圖像作為心跳
            Texture2D heartbeatTexture = new Texture2D(2, 2);
            heartbeatTexture.SetPixel(0, 0, Color.black);
            heartbeatTexture.Apply();

            // 發送心跳圖像
            await SendImageAsync(heartbeatTexture, obsSocket, obsLock, isObsConnected);

            // 接收回應 (REQ/REP模式需要)
            byte[] response = await ReceiveDataAsync(obsSocket, obsLock, isObsConnected);
            if (response == null)
            {
                Debug.LogWarning("未收到觀察心跳回應");
                isObsConnected = false;
            }

            // 銷毀臨時創建的紋理
            Destroy(heartbeatTexture);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"發送觀察心跳失敗: {e.Message}");
        }
    }

    private async void SendTopCameraHeartbeat()
    {
        if (!isTopCameraConnected || topCameraSocket == null || topCameraSocketType != ZmqSocketType.Request) return;

        try
        {
            // 創建一個1x1的空白圖像作為心跳
            Texture2D heartbeatTexture = new Texture2D(2, 2);
            heartbeatTexture.SetPixel(0, 0, Color.black);
            heartbeatTexture.Apply();

            // 發送心跳圖像
            await SendImageAsync(heartbeatTexture, topCameraSocket, topCameraLock, isTopCameraConnected);

            // 接收回應 (REQ/REP模式需要)
            byte[] response = await ReceiveDataAsync(topCameraSocket, topCameraLock, isTopCameraConnected);
            if (response == null)
            {
                Debug.LogWarning("未收到頂部相機心跳回應");
                isTopCameraConnected = false;
            }

            // 銷毀臨時創建的紋理
            Destroy(heartbeatTexture);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"發送頂部相機心跳失敗: {e.Message}");
        }
    }

    // ZeroMQ 連接信息類
    private class ConnectionInfo
    {
        public ZmqSocket Socket;
        public bool IsConnected;
        public string Host;
        public int Port;
        public object LockObj;
        public ZmqSocketType SocketType;

        public ConnectionInfo(string host, int port, object lockObj, ZmqSocketType socketType)
        {
            Host = host;
            Port = port;
            LockObj = lockObj;
            SocketType = socketType;
            IsConnected = false;
        }
    }

    // 啟動所有連接
    public async Task InitializeAllConnections()
    {
        // 創建連接信息
        var controlInfo = new ConnectionInfo(Host, ControlPort, controlLock, controlSocketType);
        var infoInfo = new ConnectionInfo(Host, InfoPort, infoLock, infoSocketType);
        var obsInfo = new ConnectionInfo(Host, ObsPort, obsLock, obsSocketType);
        var resetInfo = new ConnectionInfo(Host, ResetPort, resetLock, resetSocketType);
        var topCameraInfo = new ConnectionInfo(Host, TopCameraPort, topCameraLock, topCameraSocketType);

        // 啟動連接任務
        List<Task> connectionTasks = new List<Task>
        {
            ConnectAsync(controlInfo),
            ConnectAsync(infoInfo),
            ConnectAsync(obsInfo),
            ConnectAsync(resetInfo),
            ConnectAsync(topCameraInfo)
        };

        await Task.WhenAll(connectionTasks);

        // 更新連接狀態
        lock (controlLock) { controlSocket = controlInfo.Socket; isControlConnected = controlInfo.IsConnected; }
        lock (infoLock) { infoSocket = infoInfo.Socket; isInfoConnected = infoInfo.IsConnected; }
        lock (obsLock) { obsSocket = obsInfo.Socket; isObsConnected = obsInfo.IsConnected; }
        lock (resetLock) { resetSocket = resetInfo.Socket; isResetConnected = resetInfo.IsConnected; }
        lock (topCameraLock) { topCameraSocket = topCameraInfo.Socket; isTopCameraConnected = topCameraInfo.IsConnected; }

        Debug.Log("所有連接初始化完成");
    }

    // 連接方法
    private async Task ConnectAsync(ConnectionInfo info)
    {
        int attempts = 0;
        bool connected = false;

        while (!connected && attempts < MaxRetryAttempts)
        {
            try
            {
                Debug.Log($"嘗試連接到 {info.Host}:{info.Port}...");

                lock (info.LockObj)
                {
                    if (info.Socket != null)
                    {
                        try { info.Socket.Close(); } catch { }
                        info.Socket = null;
                    }

                    // 創建新的 ZeroMQ 套接字
                    info.Socket = new ZmqSocket(info.SocketType);

                    // 根據套接字類型選擇連接方式
                    string endpoint = $"tcp://{info.Host}:{info.Port}";

                    info.Socket.Connect(endpoint);

                    // 測試連接是否真的成功
                    if (info.SocketType == ZmqSocketType.Request)
                    {
                        // 對於 Publisher 套接字，嘗試發送一個測試消息
                        try
                        {
                            byte[] testData = Encoding.UTF8.GetBytes("{\"test\":true}");
                            bool sendResult = info.Socket.SendAsync(testData).Wait(TimeSpan.FromSeconds(connectionTestTimeout));
                            info.IsConnected = sendResult;

                            if (sendResult)
                            {
                                // 對於 Request 套接字，需要接收回應
                                try
                                {
                                    byte[] response = info.Socket.ReceiveAsync().Result;
                                    info.IsConnected = (response != null);
                                }
                                catch {
                                    info.IsConnected = false;
                                }
                            }
                            if (!sendResult)
                            {
                                Debug.LogWarning($"發送測試消息超時，連接可能失敗: {endpoint}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Debug.LogWarning($"發送測試消息失敗: {ex.Message}");
                            info.IsConnected = false;
                        }
                    }
                    else if (info.SocketType == ZmqSocketType.Request && info.Socket != null)
                    {
                        // 對於 Request 套接字，我們在實際使用時才能確定連接是否成功
                        // 暫時假設連接成功，但在第一次使用時會重新檢查
                        info.IsConnected = true;
                    }
                    else
                    {
                        // 對於其他套接字類型，暫時假設連接成功
                        info.IsConnected = true;
                    }
                    connected = true;
                }

                Debug.Log($"成功連接到 {info.Host}:{info.Port}");
            }
            catch (Exception e)
            {
                attempts++;
                Debug.LogWarning($"連接到 {info.Host}:{info.Port} 失敗 (嘗試 {attempts}/{MaxRetryAttempts}): {e.Message}");

                if (attempts < MaxRetryAttempts)
                {
                    await Task.Delay(RetryDelayMs);
                }
                else
                {
                    Debug.LogError($"連接到 {info.Host}:{info.Port} 失敗，已達最大重試次數");
                }
            }
        }
    }

    // 發送方法 (帶重試邏輯)
    public async Task<bool> SendDataAsync(byte[] data, ZmqSocket targetSocket, object lockObj, bool isConnected)
    {
        if (!isConnected || targetSocket == null) return false;

        int attempts = 0;
        bool success = false;

        while (!success && attempts < MaxRetryAttempts)
        {
            try
            {
                lock (lockObj)
                {
                    if (!isConnected || targetSocket == null) return false;

                    // 使用 ZeroMQ 發送數據
                    success = targetSocket.SendAsync(data).Result;

                    // 對於 Request 套接字，需要接收回應
                    if (success && (
                        (targetSocket == infoSocket && infoSocketType == ZmqSocketType.Request) ||
                        (targetSocket == obsSocket && obsSocketType == ZmqSocketType.Request) ||
                        (targetSocket == topCameraSocket && topCameraSocketType == ZmqSocketType.Request) ||
                        (targetSocket == controlSocket && controlSocketType == ZmqSocketType.Request)))
                    {
                        try
                        {
                            byte[] response = targetSocket.ReceiveAsync().Result;
                            success = (response != null);
                        }
                        catch (Exception e)
                        {
                            Debug.LogWarning($"接收回應失敗: {e.Message}");
                            success = false;
                        }
                    }
                    if (success)
                        Debug.Log($"成功發送 {data.Length} 字節的數據");
                }
            }
            catch (Exception e)
            {
                attempts++;
                Debug.LogWarning($"發送數據失敗 (嘗試 {attempts}/{MaxRetryAttempts}): {e.Message}");

                if (attempts < MaxRetryAttempts)
                {
                    await Task.Delay(RetryDelayMs);
                }
            }
        }

        return success;
    }

    // 接收方法 (帶重試邏輯)
    public async Task<byte[]> ReceiveDataAsync(ZmqSocket targetSocket, object lockObj, bool isConnected)
    {
        if (!isConnected || targetSocket == null) return null;

        int attempts = 0;
        byte[] result = null;

        while (result == null && attempts < MaxRetryAttempts)
        {
            try
            {
                lock (lockObj)
                {
                    if (!isConnected || targetSocket == null) return null;

                    // 使用 ZeroMQ 接收數據
                    result = targetSocket.ReceiveAsync().Result;
                }
            }
            catch (Exception e)
            {
                attempts++;
                Debug.LogWarning($"接收數據失敗 (嘗試 {attempts}/{MaxRetryAttempts}): {e.Message}");

                if (attempts < MaxRetryAttempts)
                {
                    await Task.Delay(RetryDelayMs);
                }
            }
        }

        return result;
    }

    // 發送圖像數據
    public async Task<bool> SendImageAsync(Texture2D texture, ZmqSocket targetSocket, object lockObj, bool isConnected)
    {
        if (!isConnected || targetSocket == null || texture == null) return false;

        byte[] imageBytes = texture.EncodeToPNG();
        return await SendDataAsync(imageBytes, targetSocket, lockObj, isConnected);
    }

    // 發送重置信號
    public async Task<int> SendResetSignalAsync()
    {
        Debug.Log("開始發送重置信號，檢查 Python 伺服器是否運行...");

        // 檢查連接狀態
        if (!isResetConnected)
        {
            Debug.LogWarning("重置連接未建立，嘗試初始化連接...");
            await InitializeAllConnections();

            // 再次檢查連接狀態
            if (!isResetConnected || resetSocket == null)
            {
                Debug.LogError("無法建立重置連接");
                return -1;
            }
        }

        int epoch = -1;
        int attempts = 0;
        bool success = false;
        int maxRetries = MaxRetryAttempts * 2; // 增加重試次數

        while (!success && attempts < maxRetries)
        {
            try
            {
                byte[] epochBytes = null;
                lock (resetLock)
                {
                    // 再次檢查連接狀態
                    if (!isResetConnected || resetSocket == null)
                    {
                        Debug.LogWarning("重置連接已斷開，嘗試重新連接...");
                        break; // 跳出鎖定區域，嘗試重新連接
                    }

                    Debug.Log("發送重置信號...");

                    // 發送重置信號
                    Debug.Log($"發送重置信號: 1 字節");
                    byte[] resetSignal = BitConverter.GetBytes(1);

                    // 使用超時來檢測 Python 伺服器是否真的在運行
                    var sendTask = resetSocket.SendAsync(resetSignal);
                    bool sendResult = sendTask.Wait(TimeSpan.FromSeconds(connectionTestTimeout));

                    if (!sendResult)
                    {
                        Debug.LogError("發送重置信號超時，Python 伺服器可能未運行");
                        isResetConnected = false;
                        break;
                    }

                    Debug.Log("等待接收epoch...");

                    // 在鎖內獲取套接字引用
                    var socket = resetSocket;
                }

                // 在鎖外執行異步操作
                if (resetSocket != null)
                {
                    // 接收 epoch
                    Debug.Log("等待接收epoch數據...");
                    epochBytes = await Task.Run(() => resetSocket.ReceiveAsync().Result);

                    if (epochBytes != null && epochBytes.Length == 4)
                    {
                        epoch = BitConverter.ToInt32(epochBytes, 0);
                        success = true;
                        Debug.Log($"成功接收到epoch: {epoch}");
                    }
                    else
                    {
                        Debug.LogError($"接收到的epoch數據無效: {(epochBytes != null ? epochBytes.Length : 0)} 字節");
                        Debug.LogWarning($"接收到的字節數不足: {(epochBytes != null ? epochBytes.Length : 0)}/4");
                    }
                }

                // 如果成功，直接返回
                if (success)
                {
                    return epoch;
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"發送重置信號時發生異常: {e.GetType().Name} - {e.Message}");
                Debug.LogWarning($"發送重置信號失敗 (嘗試 {attempts+1}/{maxRetries}): {e.Message}");
                // 其他異常，可能需要重新建立連接
                CloseResetConnection();
            }

            // 增加嘗試次數
            attempts++;

            // 嘗試重新建立連接
            if (!success && attempts < maxRetries)
            {
                await Task.Delay(RetryDelayMs * (attempts + 1)); // 增加延遲時間

                try
                {
                    Debug.Log($"嘗試重新建立重置連接 (嘗試 {attempts+1}/{maxRetries})...");

                    // 使用ConnectionInfo重新連接
                    var resetInfo = new ConnectionInfo(Host, ResetPort, resetLock, resetSocketType);
                    await ConnectAsync(resetInfo);

                    // 更新連接狀態
                    lock (resetLock)
                    {
                        resetSocket = resetInfo.Socket;
                        isResetConnected = resetInfo.IsConnected;
                    }

                    // 如果重新連接失敗，則繼續下一次嘗試
                    if (!isResetConnected || resetSocket == null)
                    {
                        Debug.LogWarning($"重新建立重置連接失敗 (嘗試 {attempts+1}/{maxRetries})");
                    }
                    else
                    {
                        Debug.Log("重新建立重置連接成功");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"重新連接時發生錯誤: {e.Message}");
                }
            }
        }

        if (!success)
        {
            Debug.LogError("發送重置信號失敗，已達最大重試次數");
        }

        return epoch;
    }

    // 發送控制信號
    public async Task<bool> SendControlSignalAsync(float relativeAngle)
    {
        if (!isControlConnected || controlSocket == null) return false;

        try
        {
            byte[] buffer = BitConverter.GetBytes(relativeAngle);

            Debug.Log($"準備發送控制信號: {relativeAngle}");
                bool success = false;

                if (controlSocketType == ZmqSocketType.Request)
                {
                    // 對於 Request 套接字，發送後需要接收回應
                    var sendTask = SendDataAsync(buffer, controlSocket, controlLock, isControlConnected);
                    bool sendResult = sendTask.Wait(TimeSpan.FromSeconds(connectionTestTimeout));
                    Debug.Log($"控制信號發送結果: {sendResult}");

                    if (!sendResult)
                    {
                        Debug.LogError("發送控制信號超時，Python 伺服器可能未運行");
                        return false;
                    }

                    success = await sendTask;
                    Debug.Log($"控制信號發送完成: {success}");
                }
                else
                {
                // 使用超時來檢測 Python 伺服器是否真的在運行
                var sendTask = SendDataAsync(buffer, controlSocket, controlLock, isControlConnected);
                bool sendResult = sendTask.Wait(TimeSpan.FromSeconds(connectionTestTimeout));

                if (!sendResult)
                {
                    Debug.LogError("發送控制信號超時，Python 伺服器可能未運行");
                    return false;
                }

                success = await sendTask;
                }

                return success;
        }
        catch (Exception e)
        {
            Debug.LogError($"發送控制信號時發生錯誤: {e.Message}");
            return false;
        }
    }

    // 關閉所有連接
    public void CloseAllConnections()
    {
        CloseControlConnection();
        CloseInfoConnection();
        CloseObsConnection();
        CloseResetConnection();
        CloseTopCameraConnection();
    }

    // 關閉控制連接
    private void CloseControlConnection()
    {
        lock (controlLock)
        {
            if (controlSocket != null)
            {
                try { controlSocket.Close(); } catch (Exception e) { Debug.LogError($"關閉控制套接字時發生錯誤: {e.Message}"); }
                controlSocket = null;
            }

            isControlConnected = false;
        }
    }

    // 關閉信息連接
    private void CloseInfoConnection()
    {
        lock (infoLock)
        {
            if (infoSocket != null)
            {
                try { infoSocket.Close(); } catch (Exception e) { Debug.LogError($"關閉信息套接字時發生錯誤: {e.Message}"); }
                infoSocket = null;
            }

            isInfoConnected = false;
        }
    }

    // 關閉觀察連接
    private void CloseObsConnection()
    {
        lock (obsLock)
        {
            if (obsSocket != null)
            {
                try { obsSocket.Close(); } catch (Exception e) { Debug.LogError($"關閉觀察套接字時發生錯誤: {e.Message}"); }
                obsSocket = null;
            }

            isObsConnected = false;
        }
    }

    // 關閉重置連接
    private void CloseResetConnection()
    {
        lock (resetLock)
        {
            if (resetSocket != null)
            {
                try { resetSocket.Close(); } catch (Exception e) { Debug.LogError($"關閉重置套接字時發生錯誤: {e.Message}"); }
                resetSocket = null;
            }

            isResetConnected = false;
        }
    }

    // 關閉頂部相機連接
    private void CloseTopCameraConnection()
    {
        lock (topCameraLock)
        {
            if (topCameraSocket != null)
            {
                try { topCameraSocket.Close(); } catch (Exception e) { Debug.LogError($"關閉頂部相機套接字時發生錯誤: {e.Message}"); }
                topCameraSocket = null;
            }

            isTopCameraConnected = false;
        }
    }

    // 應用退出時關閉所有連接
    private void OnApplicationQuit()
    {
        // 關閉 NetMQ
        NetMQWrapper.Instance.StopNetMQ();

        CloseAllConnections();
    }

    // 檢查 Python 伺服器是否真的在運行
    public bool IsPythonServerRunning()
    {
        // 嘗試發送重置信號，如果成功則表示 Python 伺服器在運行
        try
        {
            // 直接嘗試發送一個測試消息到重置套接字
            if (resetSocket != null && isResetConnected)
            {
                lock (resetLock)
                {
                    byte[] testData = BitConverter.GetBytes(1);
                    var sendTask = resetSocket.SendAsync(testData);
                    bool sendResult = sendTask.Wait(TimeSpan.FromSeconds(connectionTestTimeout));

                    return sendResult;
                }
            }
            return false;
        }
        catch (Exception e)
        {
            Debug.LogError($"檢查 Python 伺服器運行狀態時發生錯誤: {e.Message}");
            return false;
        }
    }
}
