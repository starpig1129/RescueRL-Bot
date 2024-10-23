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

    private void Start()
    {
        try
        {
            Debug.Log("嘗試連接到伺服器...");
            client = new TcpClient("localhost", 8000);
            client.NoDelay = true; // 禁用 Nagle 算法，避免封包延遲
            stream = client.GetStream();
            Debug.Log("成功連接到伺服器。");
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.ToString());
            this.enabled = false;
        }
    }

    private float sendInterval = 0.5f;
    private float timer = 0f;

    private void Update()
    {
        timer += Time.deltaTime;
        if (timer >= sendInterval)
        {
            Debug.Log("進入 Update 方法，開始處理資料封包...");

            Vector3 crawlerPosition = transform.position;
            Vector3 crawlerRotation = transform.up;

            GameObject[] targetObjects = GameObject.FindGameObjectsWithTag(targetTag);
            Debug.Log($"發現 {targetObjects.Length} 個標記為 {targetTag} 的目標物體。");

            // 建立一個物件來儲存要序列化的資料
            var data = new
            {
                position = new { x = crawlerPosition.x, y = crawlerPosition.y, z = crawlerPosition.z },
                rotation = new { x = crawlerRotation.x, y = crawlerRotation.y, z = crawlerRotation.z },
                targets = new List<object>()
            };

            foreach (GameObject targetObject in targetObjects)
            {
                Vector3 targetPosition = targetObject.transform.position;
                Vector3 screenPosition = crawlerCamera.WorldToScreenPoint(targetPosition);

                bool isInScreen = screenPosition.z > 0 &&
                                screenPosition.x >= 0 && screenPosition.x <= Screen.width &&
                                screenPosition.y >= 0 && screenPosition.y <= Screen.height;

                Debug.Log($"目標物位置: {targetPosition}, 是否在螢幕內: {isInScreen}");

                // 加入目標物體資料
                data.targets.Add(new
                {
                    position = new { x = targetPosition.x, y = targetPosition.y, z = targetPosition.z },
                    screenPosition = isInScreen ? new { x = (float)screenPosition.x, y = (float)screenPosition.y } : new { x = 0f, y = 0f }
                });
            }

            // 將資料序列化為 JSON 字串
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

            // 將 JSON 字串編碼為 UTF-8 位元組
            byte[] dataBytes = Encoding.UTF8.GetBytes(jsonData);

            // 獲取資料長度（僅資料，不包括長度字段），並轉換為網路位元組順序（大端序）
            int length = dataBytes.Length;
            byte[] lengthBytes = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(length));

            // 組合長度和資料
            byte[] combinedData = new byte[lengthBytes.Length + dataBytes.Length];
            Buffer.BlockCopy(lengthBytes, 0, combinedData, 0, lengthBytes.Length);
            Buffer.BlockCopy(dataBytes, 0, combinedData, lengthBytes.Length, dataBytes.Length);

            // 調試輸出
            Debug.Log($"DataBytes 長度（資料長度）: {dataBytes.Length}");
            Debug.Log($"LengthBytes 長度（長度字段）: {lengthBytes.Length}");
            Debug.Log($"CombinedData 長度（總發送長度）: {combinedData.Length}");
            Debug.Log($"發送的資料長度（length）: {length}");

            try
            {
                // 發送資料
                SendDataAsync(combinedData);
            }
            catch (Exception e)
            {
                Debug.LogError($"發送資料失敗: {e}");
            }

            timer = 0f;
        }
    }
    async void SendDataAsync(byte[] combinedData)
    {
        try
        {
            await stream.WriteAsync(combinedData, 0, combinedData.Length);
            Debug.Log("資料非同步發送成功。");
        }
        catch (Exception e)
        {
            Debug.LogError($"非同步發送資料失敗: {e}");
        }
    }

    private void OnApplicationQuit()
    {
        Debug.Log("關閉連接...");
        if (stream != null)
            stream.Close();
        if (client != null)
            client.Close();
        Debug.Log("連接已關閉。");
    }
}
