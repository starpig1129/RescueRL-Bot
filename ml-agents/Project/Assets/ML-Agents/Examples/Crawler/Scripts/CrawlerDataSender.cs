using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json; // Add this for JSON support

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
            client.NoDelay = true; // Disable Nagle's algorithm to avoid packet delays
            stream = client.GetStream();
            Debug.Log("成功連接到伺服器。");
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.ToString());
            this.enabled = false;
        }
    }

    private float sendInterval = 0.03f;
    private float timer = 0f;

    private void Update()
    {
        timer += Time.deltaTime;
        if (timer >= sendInterval)
        {
            Vector3 crawlerPosition = transform.position;
            Vector3 crawlerRotation = transform.up;
            Debug.Log($"Crawler位置: {crawlerPosition}, 旋轉: {crawlerRotation}");

            GameObject[] targetObjects = GameObject.FindGameObjectsWithTag(targetTag);
            Debug.Log($"找到 {targetObjects.Length} 個目標物體。");

            // Create an object to store data for JSON serialization
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

                // Add target data
                data.targets.Add(new
                {
                    position = new { x = targetPosition.x, y = targetPosition.y, z = targetPosition.z },
                    screenPosition = isInScreen ? new { x = (float)screenPosition.x, y = (float)screenPosition.y } : new { x = 0f, y = 0f }
                });
            }

            // Serialize data to JSON format
            string jsonData = JsonConvert.SerializeObject(data);
            Debug.Log($"生成的JSON數據: {jsonData}");

            byte[] lengthBytes = BitConverter.GetBytes(jsonData.Length);
            byte[] dataBytes = Encoding.UTF8.GetBytes(jsonData);

            byte[] combinedData = new byte[lengthBytes.Length + dataBytes.Length];
            Buffer.BlockCopy(lengthBytes, 0, combinedData, 0, lengthBytes.Length);
            Buffer.BlockCopy(dataBytes, 0, combinedData, lengthBytes.Length, dataBytes.Length);

            Debug.Log($"發送數據的字節長度: {combinedData.Length}");
            stream.Write(combinedData, 0, combinedData.Length);
            Debug.Log("數據已發送。");
            timer = 0f;
        }
    }

    private void OnApplicationQuit()
    {
        Debug.Log("關閉連接...");
        stream.Close();
        client.Close();
    }
}
