using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json;

public class CrawlerDataSender : MonoBehaviour
{
    private TcpClient client;
    private NetworkStream stream;
    public Camera crawlerCamera;
    public string targetTag = "TargetObject";
    private bool isConnected = false;

    // 優化物件池用於減少垃圾回收
    private Queue<byte[]> byteArrayPool = new Queue<byte[]>();

    private Queue<byte[]> sendQueue = new Queue<byte[]>(); // 儲存需要發送的資料
    private bool isSending = false; // 控制是否正在發送資料

    private void Start()
    {
        try
        {
            Debug.Log("嘗試連接到伺服器...");
            client = new TcpClient("localhost", 8000);
            client.NoDelay = true; // 禁用 Nagle 演算法，避免封包延遲
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("成功連接到伺服器。");
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.ToString());
            this.enabled = false;
        }
    }

    private float sendInterval = 0.01f; // 設定發送間隔
    private float timer = 0f;

    private void Update()
    {
        timer += Time.deltaTime;

        if (timer >= sendInterval && isConnected)
        {
            Debug.Log("進入 Update 方法，開始處理資料封包...");

            Vector3 crawlerPosition = transform.position;
            Vector3 crawlerRotation = transform.up;

            GameObject[] targetObjects = GameObject.FindGameObjectsWithTag(targetTag);
            Debug.Log($"發現 {targetObjects.Length} 個標記為 {targetTag} 的目標物體。");

            // 優化資料結構，減少臨時物件
            var targetDataList = new List<object>(targetObjects.Length);
            foreach (GameObject targetObject in targetObjects)
            {
                Vector3 targetPosition = targetObject.transform.position;
                Vector3 screenPosition = crawlerCamera.WorldToScreenPoint(targetPosition);

                bool isInScreen = screenPosition.z > 0 &&
                                screenPosition.x >= 0 && screenPosition.x <= Screen.width &&
                                screenPosition.y >= 0 && screenPosition.y <= Screen.height;

                Debug.Log($"目標物位置: {targetPosition}, 是否在螢幕內: {isInScreen}");

                // 加入目標物體資料到列表中
                targetDataList.Add(new
                {
                    position = new { x = targetPosition.x, y = targetPosition.y, z = targetPosition.z },
                    screenPosition = isInScreen ? new { x = (float)screenPosition.x, y = (float)screenPosition.y } : new { x = 0f, y = 0f }
                });
            }

            // 建立一個物件來儲存要序列化的資料
            var data = new
            {
                position = new { x = crawlerPosition.x, y = crawlerPosition.y, z = crawlerPosition.z },
                rotation = new { x = crawlerRotation.x, y = crawlerRotation.y, z = crawlerRotation.z },
                targets = targetDataList
            };

            // 序列化資料
            string jsonData = "";
            try
            {
                jsonData = JsonConvert.SerializeObject(data);
                Debug.Log("序列化成功。");
            }
            catch (Exception e)
            {
                Debug.LogError($"序列化失敗: {e}");
            }

            // 從物件池中獲取或創建 byte[]，避免頻繁創建新的 byte[]
            byte[] dataBytes;
            if (byteArrayPool.Count > 0)
            {
                dataBytes = byteArrayPool.Dequeue();
            }
            else
            {
                dataBytes = new byte[4096]; // 初始化一個足夠大的 byte[] 儲存資料
            }

            dataBytes = Encoding.UTF8.GetBytes(jsonData);

            // 獲取資料長度，並轉換為網路位元組順序（大端序）
            int length = dataBytes.Length;
            byte[] lengthBytes = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(length));

            // 組合長度和資料
            byte[] combinedData = new byte[lengthBytes.Length + dataBytes.Length];
            Buffer.BlockCopy(lengthBytes, 0, combinedData, 0, lengthBytes.Length);
            Buffer.BlockCopy(dataBytes, 0, combinedData, lengthBytes.Length, dataBytes.Length);

            Debug.Log($"資料準備完成。發送資料長度: {length}");

            // 加入佇列，限制同時進行的非同步發送操作
            EnqueueData(combinedData);

            // 將 dataBytes 放回物件池
            byteArrayPool.Enqueue(dataBytes);

            timer = 0f;
        }
    }

    // 排入資料佇列
    void EnqueueData(byte[] data)
    {
        sendQueue.Enqueue(data);
        if (!isSending)
            Task.Run(() => ProcessQueue());
    }

    // 處理發送佇列
    async Task ProcessQueue()
    {
        isSending = true;
        while (sendQueue.Count > 0)
        {
            byte[] dataToSend = sendQueue.Dequeue();
            try
            {
                if (stream != null && stream.CanWrite)
                {
                    await stream.WriteAsync(dataToSend, 0, dataToSend.Length);
                    await stream.FlushAsync(); // 確保資料完全寫入
                    Debug.Log("資料非同步發送成功。");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"非同步發送資料失敗: {e}");
                isConnected = false; // 如果發送失敗，設置連線狀態為無效
            }
        }
        isSending = false;
    }

    private void OnApplicationQuit()
    {
        Debug.Log("關閉連接...");
        try
        {
            if (stream != null)
                stream.Close();
            if (client != null)
                client.Close();
        }
        catch (Exception e)
        {
            Debug.LogError($"關閉連接失敗: {e}");
        }
        Debug.Log("連接已關閉。");
    }
}
