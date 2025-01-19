using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

public class CrawlerDataSender : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    public Camera crawlerCamera;
    private string targetTag = "Target";
    private bool isConnected = false;
    private float sendInterval = 0.03f; // 提高發送頻率

    // 常量定義
    private const int MAX_MESSAGE_SIZE = 1000000; // 1MB 限制，對應 Python 端限制
    private readonly object streamLock = new object();

    private bool isColliding = false;  // 碰撞狀態變數
    private float collisionTime = 0f;  // 記錄碰撞開始時間
    private bool isHoldingCollision = false; // 是否正在保持碰撞狀態
    private bool hasCollisionBeenSent = false; // 追蹤碰撞訊息是否已成功發送
    private int collisionSendAttempts = 0; // 追蹤發送嘗試次數
    private const int MAX_SEND_ATTEMPTS = 10; // 最大發送嘗試次數

    private void Start()
    {
        ConnectToServer();
        // 註冊場景加載事件
        UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private void OnDestroy()
    {
        // 取消註冊場景加載事件
        UnityEngine.SceneManagement.SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
    {
        Debug.Log($"場景已重新加載: {scene.name}");
        // 完全重置所有狀態
        ResetAllStates();
    }

    private void ResetAllStates()
    {
        // 停止所有協程
        StopAllCoroutines();

        // 重置所有碰撞相關狀態
        isColliding = false;
        isHoldingCollision = false;
        hasCollisionBeenSent = false;
        collisionSendAttempts = 0;
        collisionTime = 0f;

        // 重新初始化網路連接
        if (stream != null)
        {
            stream.Close();
            stream = null;
        }
        if (client != null)
        {
            client.Close();
            client = null;
        }

        // 重新建立連接
        ConnectToServer();

        Debug.Log("已完全重置所有狀態");
    }

    private void Update()
    {
    }

    private void ConnectToServer()
    {
        try
        {
            Debug.Log("嘗試連接到伺服器...");
            client = new TcpClient("localhost", 8000);
            client.NoDelay = true;
            client.SendBufferSize = 1024 * 64; // 增加發送緩衝區大小
            client.ReceiveBufferSize = 1024 * 64; // 增加接收緩衝區大小
            stream = client.GetStream();
            stream.WriteTimeout = 1000; // 設置寫入超時為1秒
            isConnected = true;
            Debug.Log("成功連接到伺服器。");

            InvokeRepeating("SendData", 0f, sendInterval);
        }
        catch (SocketException e)
        {
            Debug.LogError($"Socket連接異常: {e}");
            isConnected = false;
            // 5秒後重試連接
            Invoke("ConnectToServer", 5f);
        }
    }

    private void SendData()
    {
        if (!isConnected || stream == null) return;

        try
        {
            var data = PrepareData();
            string jsonData = JsonConvert.SerializeObject(data);
            byte[] dataBytes = Encoding.UTF8.GetBytes(jsonData);

            // 檢查資料大小是否超過限制
            if (dataBytes.Length > MAX_MESSAGE_SIZE)
            {
                Debug.LogError($"資料大小 ({dataBytes.Length} bytes) 超過限制 ({MAX_MESSAGE_SIZE} bytes)");
                return;
            }

            // 使用 lock 確保資料完整性
            lock (streamLock)
            {
                if (stream.CanWrite)
                {
                    // 使用大端序(Big-Endian)發送長度，對應Python端的接收方式
                    byte[] lengthBytes = BitConverter.GetBytes(dataBytes.Length);
                    if (BitConverter.IsLittleEndian)
                    {
                        Array.Reverse(lengthBytes);
                    }

                    bool sendSuccess = false;
                    try
                    {
                        // 發送數據長度
                        stream.Write(lengthBytes, 0, 4);
                        stream.Write(dataBytes, 0, dataBytes.Length);
                        stream.Flush();

                        // 立即檢查是否有確認數據
                        byte[] confirmBuffer = new byte[1];
                        if (stream.DataAvailable)
                        {
                            stream.Read(confirmBuffer, 0, 1);
                            sendSuccess = (confirmBuffer[0] == 1);
                        }
                        else
                        {
                            // 如果沒有立即收到確認，先標記為成功
                            // 讓協程繼續嘗試發送，直到收到確認或達到最大嘗試次數
                            sendSuccess = true;
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"數據發送失敗: {e.Message}");
                        sendSuccess = false;
                        HandleConnectionError();
                    }

                    // 如果正在碰撞狀態且尚未成功發送
                    if (isColliding && !hasCollisionBeenSent)
                    {
                        collisionSendAttempts++;

                        if (sendSuccess)
                        {
                            hasCollisionBeenSent = true;
                            Debug.Log($"碰撞狀態已成功發送 (第 {collisionSendAttempts} 次嘗試)");
                        }
                        else
                        {
                            Debug.LogWarning($"碰撞狀態發送失敗 (第 {collisionSendAttempts} 次嘗試)");
                            // 如果連接出現問題，觸發重連
                            if (!stream.CanWrite)
                            {
                                HandleConnectionError();
                            }
                        }
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"發送資料時發生錯誤: {e}");
            HandleConnectionError();
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            // 設置碰撞狀態
            SetCollisionState(true);
            Debug.Log($"碰撞開始: isColliding = {isColliding}, time = {collisionTime}");
        }
    }

    private void SetCollisionState(bool state)
    {
        // 停止所有現有的協程
        StopAllCoroutines();

        if (state)
        {
            isColliding = true;
            isHoldingCollision = true;
            hasCollisionBeenSent = false;
            collisionSendAttempts = 0;
            collisionTime = Time.time;

            // 安全地啟動新的碰撞保持協程
            SafeStartCoroutine(HoldCollisionState());
        }
        else
        {
            isColliding = false;
            isHoldingCollision = false;
            hasCollisionBeenSent = false;
            collisionSendAttempts = 0;
        }
    }

    private void SafeStartCoroutine(System.Collections.IEnumerator routine)
    {
        try
        {
            StartCoroutine(routine);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"協程執行錯誤: {e}");
            SetCollisionState(false);
        }
    }

    private System.Collections.IEnumerator HoldCollisionState()
    {
        float retryInterval = 0.1f; // 稍微增加重試間隔，避免過於頻繁
        float maxHoldTime = 1.0f; // 減少最大保持時間，加快重置
        float startTime = Time.time;

        while (isHoldingCollision && !hasCollisionBeenSent)
        {
            // 檢查是否超過最大保持時間
            if (Time.time - startTime > maxHoldTime)
            {
                Debug.LogWarning($"超過最大保持時間 ({maxHoldTime}秒)，結束碰撞狀態");
                break;
            }

            // 檢查是否達到最大嘗試次數
            if (collisionSendAttempts >= MAX_SEND_ATTEMPTS)
            {
                Debug.LogWarning($"已達最大嘗試次數 ({MAX_SEND_ATTEMPTS})，結束碰撞狀態");
                break;
            }

            // 檢查連接狀態
            if (!isConnected || stream == null || !stream.CanWrite)
            {
                Debug.LogWarning("連接已斷開，結束碰撞狀態");
                break;
            }

            isColliding = true;
            Debug.Log($"保持碰撞狀態... (嘗試次數: {collisionSendAttempts}, 已等待: {Time.time - startTime:F2}秒)");
            yield return new WaitForSeconds(retryInterval);
        }

        if (hasCollisionBeenSent)
        {
            Debug.Log($"碰撞訊息已成功發送，總用時: {Time.time - startTime:F2}秒");
        }
        else
        {
            Debug.LogWarning($"碰撞訊息發送失敗，總嘗試次數: {collisionSendAttempts}");
        }

        SetCollisionState(false);
    }

    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            // 在碰撞持續期間，如果尚未成功發送，保持碰撞狀態
            if (isHoldingCollision && !hasCollisionBeenSent)
            {
                isColliding = true;
            }
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            Debug.Log("碰撞物理結束，等待確認訊息發送狀態");
        }
    }

    private void OnEnable()
    {
        ResetCollisionState();
        Debug.Log("OnEnable: 重置狀態");
    }

    private void OnDisable()
    {
        ResetCollisionState();
        Debug.Log("OnDisable: 重置狀態");
    }

    private void ResetCollisionState()
    {
        StopAllCoroutines();
        isColliding = false;
        isHoldingCollision = false;
        hasCollisionBeenSent = false;
        collisionSendAttempts = 0;
        collisionTime = 0f;
        Debug.Log("ResetCollisionState: 重置碰撞狀態");
    }

    private void FixedUpdate()
    {
        // 在物理更新時輸出狀態
        if (isColliding || isHoldingCollision)
        {
            Debug.Log($"當前狀態: isColliding = {isColliding}, isHolding = {isHoldingCollision}, hasBeenSent = {hasCollisionBeenSent}, attempts = {collisionSendAttempts}");
        }
    }

    private object PrepareData()
    {
        // 使用 Transform 組件取得位置和旋轉
        Transform transform = GetComponent<Transform>();
        Vector3 crawlerPosition = transform.position;
        Vector3 crawlerRotation = transform.rotation.eulerAngles;
        GameObject[] targetObjects = GameObject.FindGameObjectsWithTag(targetTag);

        var targetDataList = new List<object>();

        foreach (GameObject targetObject in targetObjects)
        {
            Vector3 targetPosition = targetObject.transform.position;
            Vector3 screenPosition = crawlerCamera.WorldToScreenPoint(targetPosition);

            // 計算正規化的螢幕座標 (0-1 範圍)
            float normalizedX = screenPosition.z > 0 ? screenPosition.x / Screen.width : 0f;
            float normalizedY = screenPosition.z > 0 ? screenPosition.y / Screen.height : 0f;

            bool isInScreen = screenPosition.z > 0 &&
                            normalizedX >= 0 && normalizedX <= 1 &&
                            normalizedY >= 0 && normalizedY <= 1;

            targetDataList.Add(new
            {
                position = new { x = targetPosition.x, y = targetPosition.y, z = targetPosition.z },
                screenPosition = isInScreen ? new { x = normalizedX, y = normalizedY } : new { x = 0f, y = 0f }
            });
        }

        return new
        {
            position = new { x = crawlerPosition.x, y = crawlerPosition.y, z = crawlerPosition.z },
            rotation = new { x = crawlerRotation.x, y = crawlerRotation.y, z = crawlerRotation.z },
            targets = targetDataList,
            is_colliding = isColliding  // 添加碰撞狀態到傳送數據中
        };
    }

    private void HandleConnectionError()
    {
        isConnected = false;
        CancelInvoke("SendData");

        // 立即重置碰撞狀態
        SetCollisionState(false);

        if (stream != null)
        {
            stream.Close();
            stream = null;
        }

        if (client != null)
        {
            client.Close();
            client = null;
        }

        Debug.LogWarning("連接中斷，重置碰撞狀態並準備重新連接");
        // 嘗試重新連接
        Invoke("ConnectToServer", 5f);
    }

    private void OnApplicationQuit()
    {
        Debug.Log("正在關閉連接...");
        CancelInvoke("SendData");

        if (stream != null)
        {
            stream.Close();
            stream = null;
        }

        if (client != null)
        {
            client.Close();
            client = null;
        }

        Debug.Log("連接已關閉。");
    }
}
