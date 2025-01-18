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
    public string targetTag = "TargetObject";
    private bool isConnected = false;
    private float sendInterval = 0.03f; // 提高發送頻率

    // 常量定義
    private const int MAX_MESSAGE_SIZE = 1000000; // 1MB 限制，對應 Python 端限制
    private readonly object streamLock = new object();

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

    private bool isSceneReloading = false;  // 新增場景重載標記

    private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
    {
        Debug.Log($"場景已重新加載: {scene.name}, 保持碰撞狀態: {isColliding}");
        isSceneReloading = true;
        SafeStartCoroutine(HandleSceneLoadedState());
    }

    private System.Collections.IEnumerator HandleSceneLoadedState()
    {
        // 等待兩幀確保所有物件都已初始化
        yield return null;
        yield return null;

        if (isColliding || isHoldingCollision)
        {
            // 確保在場景重載後狀態保持
            DontDestroyOnLoad(gameObject);

            // 強制更新狀態
            isColliding = true;
            isHoldingCollision = true;
            collisionTime = Time.time;  // 重置碰撞時間

            // 重新啟動碰撞保持協程
            StopAllCoroutines();
            SafeStartCoroutine(HoldCollisionState());

            Debug.Log($"場景重載後強制更新狀態: isColliding = {isColliding}, isHolding = {isHoldingCollision}");
        }

        // 等待一小段時間後再清除重載標記
        yield return new WaitForSeconds(0.5f);
        isSceneReloading = false;
        Debug.Log("場景重載處理完成");
    }

    private void Update()
    {
        // 在場景重載過程中強制保持碰撞狀態
        if (isSceneReloading && (isColliding || isHoldingCollision))
        {
            isColliding = true;
            isHoldingCollision = true;
        }
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

            // 如果正在場景重載且有碰撞狀態，強制設置碰撞標記
            if (isSceneReloading && (isColliding || isHoldingCollision))
            {
                var originalData = (dynamic)data;
                data = new
                {
                    position = originalData.position,
                    rotation = originalData.rotation,
                    targets = originalData.targets,
                    is_colliding = true  // 強制設置為 true
                };
            }

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

                    stream.Write(lengthBytes, 0, 4);
                    stream.Write(dataBytes, 0, dataBytes.Length);
                    stream.Flush();

                    // 只在狀態變化時輸出日誌
                    if (isColliding)
                    {
                        Debug.Log($"發送碰撞狀態: isColliding = {isColliding}");
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

    private bool isColliding = false;  // 碰撞狀態變數
    private float collisionTime = 0f;  // 記錄碰撞開始時間
    private const float MIN_COLLISION_TIME = 1.5f; // 增加最小碰撞持續時間，確保狀態能被傳送
    private const float COLLISION_HOLD_TIME = 2.0f; // 延長保持時間，確保能覆蓋場景重載過程
    private bool isHoldingCollision = false; // 是否正在保持碰撞狀態

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
            collisionTime = Time.time;

            // 使用 DontDestroyOnLoad 確保物件在場景重載時不被銷毀
            DontDestroyOnLoad(gameObject);

            // 安全地啟動新的碰撞保持協程
            SafeStartCoroutine(HoldCollisionState());
        }
        else
        {
            isColliding = false;
            isHoldingCollision = false;
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
        yield return new WaitForSeconds(MIN_COLLISION_TIME);
        Debug.Log($"最小碰撞時間已過: {MIN_COLLISION_TIME}秒");

        float holdEndTime = Time.time + COLLISION_HOLD_TIME;
        while (Time.time < holdEndTime && isHoldingCollision)
        {
            isColliding = true;
            Debug.Log($"保持碰撞狀態: 剩餘 {holdEndTime - Time.time:F2} 秒");
            yield return new WaitForSeconds(0.1f);
        }

        Debug.Log("碰撞保持時間結束");
        SetCollisionState(false);
    }

    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            // 在碰撞持續期間保持狀態
            if (isHoldingCollision)
            {
                isColliding = true;
            }
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            // 不立即重置碰撞狀態，讓協程處理
            Debug.Log($"碰撞物理結束，等待保持時間結束");
        }
    }

    private void OnEnable()
    {
        if (!isSceneReloading)
        {
            ResetCollisionState();
            Debug.Log("OnEnable: 正常重置狀態");
        }
        else
        {
            Debug.Log("OnEnable: 場景重載中，跳過重置");
        }
    }

    private void OnDisable()
    {
        if (!isSceneReloading)
        {
            ResetCollisionState();
            Debug.Log("OnDisable: 正常重置狀態");
        }
        else
        {
            // 在場景重載時保持狀態
            if (isColliding || isHoldingCollision)
            {
                DontDestroyOnLoad(gameObject);
                Debug.Log("OnDisable: 場景重載中，保持碰撞狀態");
            }
        }
    }

    private void ResetCollisionState()
    {
        // 如果正在場景重載且有碰撞狀態，則不重置
        if (isSceneReloading && (isColliding || isHoldingCollision))
        {
            Debug.Log("ResetCollisionState: 場景重載中，保持碰撞狀態");
            return;
        }

        StopAllCoroutines();
        isColliding = false;
        isHoldingCollision = false;
        collisionTime = 0f;
        Debug.Log("ResetCollisionState: 完全重置狀態");
    }

    private void FixedUpdate()
    {
        // 在物理更新時輸出狀態
        if (isColliding || isHoldingCollision)
        {
            Debug.Log($"當前狀態: isColliding = {isColliding}, isHolding = {isHoldingCollision}, time since collision = {Time.time - collisionTime}");
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
