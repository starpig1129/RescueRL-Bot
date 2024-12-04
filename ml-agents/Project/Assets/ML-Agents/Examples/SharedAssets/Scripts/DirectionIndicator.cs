using UnityEngine;
using System.Net.Sockets;
using System;
using System.Threading;

namespace Unity.MLAgentsExamples
{
    public class DirectionIndicator : MonoBehaviour
    {
        public bool updatedByAgent;
        public Transform transformToFollow;  // Crawler的Transform
        public Transform targetToFollow;     // 要跟隨的目標物件
        public float heightOffset;
        public float magnitude = 2.0f;  // 指示器與Crawler的距離

        private float m_StartingYPos;
        private TcpClient tcpClient;
        private NetworkStream stream;
        private Thread tcpThread;

        private float relativeAngle = 0f;  // 相對角度
        private bool dataReceived;

        void Start()
        {
            // 連接到 Python 伺服器
            tcpClient = new TcpClient("localhost", 5000);
            stream = tcpClient.GetStream();
            tcpThread = new Thread(new ThreadStart(ReadData));
            tcpThread.IsBackground = true;
            tcpThread.Start();

            // 初始化起始位置
            m_StartingYPos = transform.position.y;
        }

        void ReadData()
        {
            try
            {
                while (tcpClient.Connected)
                {
                    byte[] buffer = new byte[4];  // 只接收一個float (相對角度)
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 4)
                    {
                        // 接收從Python發送的相對角度(弧度)
                        relativeAngle = BitConverter.ToSingle(buffer, 0) * Mathf.Rad2Deg;
                        dataReceived = true;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError("TCP Read Error: " + e.Message);
            }
        }

        void Update()
        {
            if (updatedByAgent || !dataReceived)
                return;

            // 獲取Crawler的當前朝向
            float crawlerRotation = transformToFollow.eulerAngles.y;

            // 計算世界空間中的目標角度
            float worldAngle = crawlerRotation + relativeAngle;

            // 將角度轉換為弧度
            float angleRad = worldAngle * Mathf.Deg2Rad;

            // 計算相對於Crawler的位置
            float targetX = magnitude * Mathf.Sin(angleRad);
            float targetZ = magnitude * Mathf.Cos(angleRad);

            // 計算新位置
            Vector3 newPosition = transformToFollow.position + new Vector3(
                targetX,
                heightOffset,
                targetZ
            );

            // 平滑移動到新位置
            transform.position = Vector3.Lerp(
                transform.position,
                newPosition,
                Time.deltaTime * 5f
            );

            // 更新朝向 - 始終朝向移動方向
            Vector3 direction = new Vector3(
                Mathf.Sin(angleRad),
                0f,
                Mathf.Cos(angleRad)
            );

            // 平滑旋轉
            Quaternion targetRotation = Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                targetRotation,
                Time.deltaTime * 5f
            );

            dataReceived = false;
        }

        void OnDisable()
        {
            // 關閉TCP連接
            if (tcpThread != null && tcpThread.IsAlive)
                tcpThread.Abort();

            stream?.Close();
            tcpClient?.Close();
        }

        public void MatchOrientation(Transform t)
        {
            if (!updatedByAgent && dataReceived)
            {
                float crawlerRotation = transformToFollow.eulerAngles.y;
                float worldAngle = crawlerRotation + relativeAngle;
                float angleRad = worldAngle * Mathf.Deg2Rad;

                float targetX = magnitude * Mathf.Sin(angleRad);
                float targetZ = magnitude * Mathf.Cos(angleRad);

                // 立即更新位置
                Vector3 newPosition = transformToFollow.position + new Vector3(
                    targetX,
                    heightOffset,
                    targetZ
                );

                transform.position = newPosition;
                transform.rotation = Quaternion.LookRotation(new Vector3(
                    Mathf.Sin(angleRad),
                    0f,
                    Mathf.Cos(angleRad)
                ));
            }
        }
    }
}
