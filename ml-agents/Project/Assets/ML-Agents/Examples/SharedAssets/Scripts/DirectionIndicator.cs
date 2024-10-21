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

        private float targetFordX;
        private float targetFordZ;
        private bool dataReceived;

        void Start()
        {
            // 连接到 Python 服务器
            tcpClient = new TcpClient("localhost", 5000); // 替换为服务器 IP 和端口
            stream = tcpClient.GetStream();
            tcpThread = new Thread(new ThreadStart(ReadData));
            tcpThread.IsBackground = true;
            tcpThread.Start();
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
                        targetFordX = BitConverter.ToSingle(buffer, 0);
                        targetFordZ = BitConverter.ToSingle(buffer, 4);
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

            // 更新位置
            transform.position = new Vector3(transformToFollow.position.x + targetFordX, m_StartingYPos + heightOffset, transformToFollow.position.z + targetFordZ);
            Vector3 walkDir = targetToLookAt.position + transform.position;
            walkDir.y = 0;
            transform.rotation = Quaternion.LookRotation(walkDir);

            dataReceived = false; // Reset flag after updating position
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
            // 更新位置至指定Transform的位置，加上高度偏移
            transform.position = new Vector3(t.position.x, m_StartingYPos + heightOffset, t.position.z);
            // 將朝向設定為指定Transform的朝向
            transform.rotation = t.rotation;
        }
    }
}
