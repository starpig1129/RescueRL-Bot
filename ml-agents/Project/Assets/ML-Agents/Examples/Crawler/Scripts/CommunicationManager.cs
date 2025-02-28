using UnityEngine;
using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;
using System.IO;

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

    // 各種連接
    private TcpClient controlClient; // 端口5000
    private TcpClient infoClient;    // 端口8000
    private TcpClient obsClient;     // 端口6000
    private TcpClient resetClient;   // 端口7000
    private TcpClient topCameraClient; // 端口9000

    // 對應的網絡流
    private NetworkStream controlStream;
    private NetworkStream infoStream;
    private NetworkStream obsStream;
    private NetworkStream resetStream;
    private NetworkStream topCameraStream;

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
    private const string Host = "localhost";
    private const int ControlPort = 5000;
    private const int InfoPort = 8000;
    private const int ObsPort = 6000;
    private const int ResetPort = 7000;
    private const int TopCameraPort = 9000;

    // 重連參數
    private const int MaxRetryAttempts = 5;
    private const int RetryDelayMs = 1000;

    // 公開屬性
    public bool IsControlConnected => isControlConnected;
    public bool IsInfoConnected => isInfoConnected;
    public bool IsObsConnected => isObsConnected;
    public bool IsResetConnected => isResetConnected;
    public bool IsTopCameraConnected => isTopCameraConnected;

    public NetworkStream ControlStream => controlStream;
    public NetworkStream InfoStream => infoStream;
    public NetworkStream ObsStream => obsStream;
    public NetworkStream ResetStream => resetStream;
    public NetworkStream TopCameraStream => topCameraStream;

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

    private System.Collections.IEnumerator MonitorConnections()
    {
        while (isMonitoringConnections)
        {
            yield return new WaitForSeconds(connectionCheckInterval);

            // 檢查所有連接
            bool needReconnect = false;

            if (!isControlConnected || controlStream == null) needReconnect = true;
            if (!isInfoConnected || infoStream == null) needReconnect = true;
            if (!isObsConnected || obsStream == null) needReconnect = true;
            if (!isResetConnected || resetStream == null) needReconnect = true;
            if (!isTopCameraConnected || topCameraStream == null) needReconnect = true;

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

    // 連接信息類
    private class ConnectionInfo
    {
        public TcpClient Client;
        public NetworkStream Stream;
        public bool IsConnected;
        public string Host;
        public int Port;
        public object LockObj;

        public ConnectionInfo(string host, int port, object lockObj)
        {
            Host = host;
            Port = port;
            LockObj = lockObj;
            IsConnected = false;
        }
    }

    // 啟動所有連接
    public async Task InitializeAllConnections()
    {
        // 創建連接信息
        var controlInfo = new ConnectionInfo(Host, ControlPort, controlLock);
        var infoInfo = new ConnectionInfo(Host, InfoPort, infoLock);
        var obsInfo = new ConnectionInfo(Host, ObsPort, obsLock);
        var resetInfo = new ConnectionInfo(Host, ResetPort, resetLock);
        var topCameraInfo = new ConnectionInfo(Host, TopCameraPort, topCameraLock);

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
        lock (controlLock) { controlClient = controlInfo.Client; controlStream = controlInfo.Stream; isControlConnected = controlInfo.IsConnected; }
        lock (infoLock) { infoClient = infoInfo.Client; infoStream = infoInfo.Stream; isInfoConnected = infoInfo.IsConnected; }
        lock (obsLock) { obsClient = obsInfo.Client; obsStream = obsInfo.Stream; isObsConnected = obsInfo.IsConnected; }
        lock (resetLock) { resetClient = resetInfo.Client; resetStream = resetInfo.Stream; isResetConnected = resetInfo.IsConnected; }
        lock (topCameraLock) { topCameraClient = topCameraInfo.Client; topCameraStream = topCameraInfo.Stream; isTopCameraConnected = topCameraInfo.IsConnected; }

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
                    if (info.Client != null)
                    {
                        try { info.Client.Close(); } catch { }
                        info.Client = null;
                    }

                    info.Client = new TcpClient();
                    info.Client.NoDelay = true;
                    info.Client.SendBufferSize = 1024 * 256;
                    info.Client.ReceiveBufferSize = 1024 * 256;
                }

                await info.Client.ConnectAsync(info.Host, info.Port);

                lock (info.LockObj)
                {
                    info.Stream = info.Client.GetStream();
                    info.IsConnected = true;
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
    public async Task<bool> SendDataAsync(byte[] data, NetworkStream targetStream, object lockObj, bool isConnected)
    {
        if (!isConnected || targetStream == null) return false;

        int attempts = 0;
        bool success = false;

        while (!success && attempts < MaxRetryAttempts)
        {
            try
            {
                lock (lockObj)
                {
                    if (!isConnected || targetStream == null) return false;

                    // 發送數據長度
                    byte[] lengthBytes = BitConverter.GetBytes(data.Length);
                    targetStream.Write(lengthBytes, 0, 4);

                    // 發送數據
                    targetStream.Write(data, 0, data.Length);
                    targetStream.Flush();

                    success = true;
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
    public async Task<byte[]> ReceiveDataAsync(NetworkStream targetStream, object lockObj, bool isConnected)
    {
        if (!isConnected || targetStream == null) return null;

        int attempts = 0;
        byte[] result = null;

        while (result == null && attempts < MaxRetryAttempts)
        {
            try
            {
                lock (lockObj)
                {
                    if (!isConnected || targetStream == null) return null;

                    // 接收數據長度
                    byte[] lengthBytes = new byte[4];
                    int bytesRead = targetStream.Read(lengthBytes, 0, 4);

                    if (bytesRead != 4) throw new Exception("無法讀取數據長度");

                    int length = BitConverter.ToInt32(lengthBytes, 0);

                    // 接收數據
                    byte[] data = new byte[length];
                    int totalBytesRead = 0;

                    while (totalBytesRead < length)
                    {
                        int bytesRemaining = length - totalBytesRead;
                        int bytesReadThisTime = targetStream.Read(data, totalBytesRead, bytesRemaining);

                        if (bytesReadThisTime == 0) throw new Exception("連接已關閉");

                        totalBytesRead += bytesReadThisTime;
                    }

                    result = data;
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
    public async Task<bool> SendImageAsync(Texture2D texture, NetworkStream targetStream, object lockObj, bool isConnected)
    {
        if (!isConnected || targetStream == null || texture == null) return false;

        byte[] imageBytes = texture.EncodeToJPG();
        return await SendDataAsync(imageBytes, targetStream, lockObj, isConnected);
    }

    // 發送重置信號
    public async Task<int> SendResetSignalAsync()
    {
        Debug.Log("開始發送重置信號...");

        // 檢查連接狀態
        if (!isResetConnected)
        {
            Debug.LogWarning("重置連接未建立，嘗試初始化連接...");
            await InitializeAllConnections();

            // 再次檢查連接狀態
            if (!isResetConnected || resetStream == null)
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
                lock (resetLock)
                {
                    // 再次檢查連接狀態和流狀態
                    if (!isResetConnected || resetStream == null)
                    {
                        Debug.LogWarning("重置連接已斷開，嘗試重新連接...");
                        break; // 跳出鎖定區域，嘗試重新連接
                    }

                    // 檢查流是否可寫入和讀取
                    if (!resetStream.CanWrite || !resetStream.CanRead)
                    {
                        Debug.LogWarning("重置流無法讀寫，嘗試重新連接...");
                        break; // 跳出鎖定區域，嘗試重新連接
                    }

                    Debug.Log("發送重置信號...");

                    // 發送重置信號
                    byte[] resetSignal = BitConverter.GetBytes(1);
                    resetStream.Write(resetSignal, 0, 4);
                    resetStream.Flush(); // 確保數據立即發送

                    Debug.Log("等待接收epoch...");

                    // 設置讀取超時
                    resetStream.ReadTimeout = 5000; // 5秒超時

                    // 接收 epoch
                    byte[] epochBytes = new byte[4];
                    int bytesRead = resetStream.Read(epochBytes, 0, 4);

                    if (bytesRead == 4)
                    {
                        epoch = BitConverter.ToInt32(epochBytes, 0);
                        success = true;
                        Debug.Log($"成功接收到epoch: {epoch}");
                    }
                    else
                    {
                        Debug.LogWarning($"接收到的字節數不足: {bytesRead}/4");
                    }
                }

                // 如果成功，直接返回
                if (success)
                {
                    return epoch;
                }
            }
            catch (ObjectDisposedException ode)
            {
                Debug.LogError($"重置流已被釋放: {ode.Message}");
                // 嘗試重新建立連接
                CloseResetConnection();
            }
            catch (IOException ioe)
            {
                Debug.LogWarning($"IO異常: {ioe.Message}");
                // 可能是連接已斷開，嘗試重新建立連接
                CloseResetConnection();
            }
            catch (Exception e)
            {
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
                    var resetInfo = new ConnectionInfo(Host, ResetPort, resetLock);
                    await ConnectAsync(resetInfo);

                    // 更新連接狀態
                    lock (resetLock)
                    {
                        resetClient = resetInfo.Client;
                        resetStream = resetInfo.Stream;
                        isResetConnected = resetInfo.IsConnected;
                    }

                    // 如果重新連接失敗，則繼續下一次嘗試
                    if (!isResetConnected || resetStream == null)
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
        if (!isControlConnected || controlStream == null) return false;

        byte[] buffer = BitConverter.GetBytes(relativeAngle);
        return await SendDataAsync(buffer, controlStream, controlLock, isControlConnected);
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
            if (controlStream != null)
            {
                try { controlStream.Close(); } catch (Exception e) { Debug.LogError($"關閉控制流時發生錯誤: {e.Message}"); }
                controlStream = null;
            }

            if (controlClient != null)
            {
                try { controlClient.Close(); } catch (Exception e) { Debug.LogError($"關閉控制客戶端時發生錯誤: {e.Message}"); }
                controlClient = null;
            }

            isControlConnected = false;
        }
    }

    // 關閉信息連接
    private void CloseInfoConnection()
    {
        lock (infoLock)
        {
            if (infoStream != null)
            {
                try { infoStream.Close(); } catch (Exception e) { Debug.LogError($"關閉信息流時發生錯誤: {e.Message}"); }
                infoStream = null;
            }

            if (infoClient != null)
            {
                try { infoClient.Close(); } catch (Exception e) { Debug.LogError($"關閉信息客戶端時發生錯誤: {e.Message}"); }
                infoClient = null;
            }

            isInfoConnected = false;
        }
    }

    // 關閉觀察連接
    private void CloseObsConnection()
    {
        lock (obsLock)
        {
            if (obsStream != null)
            {
                try { obsStream.Close(); } catch (Exception e) { Debug.LogError($"關閉觀察流時發生錯誤: {e.Message}"); }
                obsStream = null;
            }

            if (obsClient != null)
            {
                try { obsClient.Close(); } catch (Exception e) { Debug.LogError($"關閉觀察客戶端時發生錯誤: {e.Message}"); }
                obsClient = null;
            }

            isObsConnected = false;
        }
    }

    // 關閉重置連接
    private void CloseResetConnection()
    {
        lock (resetLock)
        {
            if (resetStream != null)
            {
                try { resetStream.Close(); } catch (Exception e) { Debug.LogError($"關閉重置流時發生錯誤: {e.Message}"); }
                resetStream = null;
            }

            if (resetClient != null)
            {
                try { resetClient.Close(); } catch (Exception e) { Debug.LogError($"關閉重置客戶端時發生錯誤: {e.Message}"); }
                resetClient = null;
            }

            isResetConnected = false;
        }
    }

    // 關閉頂部相機連接
    private void CloseTopCameraConnection()
    {
        lock (topCameraLock)
        {
            if (topCameraStream != null)
            {
                try { topCameraStream.Close(); } catch (Exception e) { Debug.LogError($"關閉頂部相機流時發生錯誤: {e.Message}"); }
                topCameraStream = null;
            }

            if (topCameraClient != null)
            {
                try { topCameraClient.Close(); } catch (Exception e) { Debug.LogError($"關閉頂部相機客戶端時發生錯誤: {e.Message}"); }
                topCameraClient = null;
            }

            isTopCameraConnected = false;
        }
    }

    // 應用退出時關閉所有連接
    private void OnApplicationQuit()
    {
        CloseAllConnections();
    }
}
