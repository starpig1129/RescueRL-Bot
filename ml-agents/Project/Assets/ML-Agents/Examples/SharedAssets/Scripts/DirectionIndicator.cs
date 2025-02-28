using UnityEngine;
using System;
using System.Threading.Tasks;

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
        private bool isRunning = true;
        private float relativeAngle = 0f;  // 相對角度
        private bool dataReceived;
        private object lockObject = new object();
        private float receiveInterval = 0.01f;

        void Start()
        {
            // 初始化起始位置
            m_StartingYPos = transform.position.y;

            // 確保CommunicationManager實例存在
            var manager = CommunicationManager.Instance;

            // 開始接收數據的協程
            StartCoroutine(ReceiveDataRoutine());
        }

        private System.Collections.IEnumerator ReceiveDataRoutine()
        {
            while (isRunning)
            {
                yield return new WaitForSeconds(receiveInterval);

                Debug.Log($"開始接收控制訊號");

                // 使用通信管理器接收數據
                ReceiveControlSignalAsync().ContinueWith(task => {
                    if (task.Exception != null)
                    {
                        Debug.LogError($"接收控制信號時發生錯誤: {task.Exception.Message}");
                    }
                });
            }
        }

        private async Task ReceiveControlSignalAsync()
        {
            try
            {
                // 使用通信管理器接收數據
                byte[] data = await CommunicationManager.Instance.ReceiveDataAsync(
                    CommunicationManager.Instance.ControlStream,
                    CommunicationManager.Instance.ControlLock,
                    CommunicationManager.Instance.IsControlConnected
                );

                if (data != null && data.Length == 4)
                {
                    lock (lockObject)
                    {
                        relativeAngle = BitConverter.ToSingle(data, 0) * Mathf.Rad2Deg;
                        dataReceived = true;
                    }
                    Debug.Log($"控制訊號成功");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"接收控制信號時發生錯誤: {e.Message}");
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
            isRunning = false;
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
