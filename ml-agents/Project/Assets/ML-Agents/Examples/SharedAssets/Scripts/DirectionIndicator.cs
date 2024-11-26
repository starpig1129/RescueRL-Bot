using UnityEngine;
using System.Net.Sockets;
using System;
using System.Threading;

namespace Unity.MLAgentsExamples
{
    public class DirectionIndicator : MonoBehaviour
    {
        public bool updatedByAgent;
        public Transform transformToFollow;
        public Transform targetToLookAt;
        public float heightOffset;

        private float m_StartingYPos;
        private TcpClient tcpClient;
        private NetworkStream stream;
        private Thread tcpThread;

        private float currentAngle = 0f;    // 當前絕對角度
        private float targetFordX;          // 目標前進方向X
        private float targetFordZ;          // 目標前進方向Z
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
                    byte[] buffer = new byte[8];
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 8)
                    {
                        // 接收從Python發送的方向向量
                        targetFordX = BitConverter.ToSingle(buffer, 0) * 2;
                        targetFordZ = BitConverter.ToSingle(buffer, 4) * 2;

                        // 計算新的絕對角度
                        currentAngle = Mathf.Atan2(targetFordZ, targetFordX) * Mathf.Rad2Deg;
                        if (currentAngle < 0)
                            currentAngle += 360f;

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

            // 計算新位置 - 相對於Crawler的位置
            Vector3 newPosition = transformToFollow.position + new Vector3(
                targetFordX,
                heightOffset,
                targetFordZ
            );

            // 平滑移動到新位置
            transform.position = Vector3.Lerp(
                transform.position,
                newPosition,
                Time.deltaTime * 5f  // 調整移動速度
            );

            // 更新朝向 - 使用當前角度
            Vector3 direction = new Vector3(
                Mathf.Cos(currentAngle * Mathf.Deg2Rad),
                0f,
                Mathf.Sin(currentAngle * Mathf.Deg2Rad)
            );

            // 平滑旋轉
            Quaternion targetRotation = Quaternion.LookRotation(direction);
            transform.rotation = Quaternion.Slerp(
                transform.rotation,
                targetRotation,
                Time.deltaTime * 5f  // 調整旋轉速度
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
                // 計算新位置
                Vector3 newPosition = transformToFollow.position + new Vector3(
                    targetFordX,
                    heightOffset,
                    targetFordZ
                );

                // 立即更新位置和朝向
                transform.position = newPosition;
                transform.rotation = Quaternion.LookRotation(new Vector3(
                    Mathf.Cos(currentAngle * Mathf.Deg2Rad),
                    0f,
                    Mathf.Sin(currentAngle * Mathf.Deg2Rad)
                ));
            }
        }

        // 獲取當前角度（用於調試）
        public float GetCurrentAngle()
        {
            return currentAngle;
        }
    }
}
