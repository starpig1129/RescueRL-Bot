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

    // 優化物件池用於減少垃圾回收
    private Queue<byte[]> byteArrayPool = new Queue<byte[]>();
    private float sendInterval = 0.03f; // 改為與camera相同的傳輸間隔

    private void Start()
    {
        try
        {
            Debug.Log("嘗試連接到伺服器...");
            client = new TcpClient("localhost", 8000);
            client.NoDelay = true;
            stream = client.GetStream();
            isConnected = true;
            Debug.Log("成功連接到伺服器。");

            // 使用InvokeRepeating替代Update中的計時器
            InvokeRepeating("SendData", 0f, sendInterval);
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.ToString());
            this.enabled = false;
        }
    }

    private void SendData()
    {
        if (!isConnected || stream == null) return;

        Debug.Log("開始處理資料封包...");

        Vector3 crawlerPosition = transform.position;
        Vector3 crawlerRotation = transform.up;

        GameObject[] targetObjects = GameObject.FindGameObjectsWithTag(targetTag);
        Debug.Log($"發現 {targetObjects.Length} 個標記為 {targetTag} 的目標物體。");

        var targetDataList = new List<object>(targetObjects.Length);
        foreach (GameObject targetObject in targetObjects)
        {
            Vector3 targetPosition = targetObject.transform.position;
            Vector3 screenPosition = crawlerCamera.WorldToScreenPoint(targetPosition);

            bool isInScreen = screenPosition.z > 0 &&
                            screenPosition.x >= 0 && screenPosition.x <= Screen.width &&
                            screenPosition.y >= 0 && screenPosition.y <= Screen.height;

            targetDataList.Add(new
            {
                position = new { x = targetPosition.x, y = targetPosition.y, z = targetPosition.z },
                screenPosition = isInScreen ? new { x = (float)screenPosition.x, y = (float)screenPosition.y } : new { x = 0f, y = 0f }
            });
        }

        var data = new
        {
            position = new { x = crawlerPosition.x, y = crawlerPosition.y, z = crawlerPosition.z },
            rotation = new { x = crawlerRotation.x, y = crawlerRotation.y, z = crawlerRotation.z },
            targets = targetDataList
        };

        try
        {
            string jsonData = JsonConvert.SerializeObject(data);
            byte[] dataBytes = Encoding.UTF8.GetBytes(jsonData);

            // 直接發送資料長度和資料本身
            try
            {
                if (stream.CanWrite)
                {
                    // 先發送資料長度
                    byte[] lengthBytes = BitConverter.GetBytes(dataBytes.Length);
                    stream.Write(lengthBytes, 0, 4);
                    // 再發送資料本身
                    stream.Write(dataBytes, 0, dataBytes.Length);
                    stream.Flush();
                    Debug.Log($"成功發送資料，長度: {dataBytes.Length}");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"發送資料失敗: {e}");
                isConnected = false;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"序列化失敗: {e}");
        }
    }

    private void OnApplicationQuit()
    {
        Debug.Log("關閉連接...");
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }

        // 取消定期發送
        CancelInvoke("SendData");
        Debug.Log("連接已關閉。");
    }
}
