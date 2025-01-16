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
    public string targetTag = "target"; // 改為檢測 target tag
    private bool isConnected = false;
    private float sendInterval = 0.03f;

    // 常量定義
    private const int MAX_MESSAGE_SIZE = 1000000; // 1MB 限制，對應 Python 端限制
    private readonly object streamLock = new object();

    private void Start()
    {
        ConnectToServer();
    }

    private void ConnectToServer()
    {
        try
        {
            Debug.Log("嘗試連接到伺服器...");
            client = new TcpClient("localhost", 8000);
            client.NoDelay = true;
            stream = client.GetStream();
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

                    stream.Write(lengthBytes, 0, 4);
                    stream.Write(dataBytes, 0, dataBytes.Length);
                    stream.Flush();

                    Debug.Log($"成功發送資料，長度: {dataBytes.Length} bytes");
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
    private const float MIN_COLLISION_TIME = 0.1f; // 最小碰撞持續時間

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            isColliding = true;
            collisionTime = Time.time;
            Debug.Log($"碰撞開始: isColliding = {isColliding}, time = {collisionTime}");
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            // 確保碰撞狀態持續一段時間
            if (Time.time - collisionTime >= MIN_COLLISION_TIME)
            {
                isColliding = true;
            }
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.CompareTag(targetTag))
        {
            isColliding = false;
            Debug.Log($"碰撞結束: isColliding = {isColliding}, duration = {Time.time - collisionTime}");
        }
    }

    private void OnEnable()
    {
        isColliding = false;
        collisionTime = 0f;
    }

    private void OnDisable()
    {
        isColliding = false;
        collisionTime = 0f;
    }

    private void FixedUpdate()
    {
        // 在物理更新時檢查並更新碰撞狀態
        if (isColliding && Time.time - collisionTime < MIN_COLLISION_TIME)
        {
            Debug.Log($"保持碰撞狀態: duration = {Time.time - collisionTime}");
        }
    }

    private object PrepareData()
    {
        // 獲取 Rigidbody 組件以取得精確的物理位置和旋轉
        Rigidbody rb = GetComponent<Rigidbody>();
        Vector3 crawlerPosition = rb.position;
        Vector3 crawlerRotation = rb.rotation.eulerAngles;
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
