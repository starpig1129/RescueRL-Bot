using UnityEngine;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using UnityEngine.Events;
using UnityEngine.SceneManagement;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Threading.Tasks;
using System.Threading;
using System.Diagnostics;
using System.Collections;
using System.Security.Authentication.ExtendedProtection;

namespace Unity.MLAgentsExamples
{
    /// <summary>
    /// 用於目標放置和與代理碰撞檢測的實用類。
    /// 將此腳本添加到您希望代理觸摸的目標上。
    /// 每當使用標記為'tagToDetect'的碰撞器觸摸目標時，將觸發回調。
    /// </summary>
    public class TargetController : MonoBehaviour
    {

        [Header("檢測的碰撞器標記")]
        public string tagToDetect = "Crawler"; //要檢測的碰撞器標記

        [Header("目標放置")]
        public float spawnRadius; //隨機生成目標的半徑。
        public bool respawnIfTouched; //當觸摸到目標時，是否重新生成目標。

        [Header("目標墜落保護")]
        public bool respawnIfFallsOffPlatform = true; //如果目標從平台上掉落，則重置位置。
        public float fallDistance = 5; //觸發重新生成的下墜距離

        private Vector3 m_startingPos; //目標的初始位置

        [System.Serializable]
        public class TriggerEvent : UnityEvent<Collider>
        {
        }

        [Header("觸發器回調")]
        public TriggerEvent onTriggerEnterEvent = new TriggerEvent();
        public TriggerEvent onTriggerStayEvent = new TriggerEvent();
        public TriggerEvent onTriggerExitEvent = new TriggerEvent();

        [System.Serializable]
        public class CollisionEvent : UnityEvent<Collision>
        {
        }

        [Header("碰撞器回調")]
        public CollisionEvent onCollisionEnterEvent = new CollisionEvent();
        public CollisionEvent onCollisionStayEvent = new CollisionEvent();
        public CollisionEvent onCollisionExitEvent = new CollisionEvent();

        // Start is called before the first frame update
        void OnEnable()
        {
            m_startingPos = transform.position;
            if (respawnIfTouched)
            {
                //MoveTargetToRandomPosition();
            }
        }

        void Update()
        {
            if (respawnIfFallsOffPlatform)
            {
                if (transform.position.y < m_startingPos.y - fallDistance)
                {
                    //UnityEngine.Debug.Log($"{transform.name} 從平台上掉落了");
                    //MoveTargetToRandomPosition();
                }
            }
        }

        //新增
        public Vector3[] targetPositions; // 設置目標位置
        public static int num; // 設置靜態變數 'num'，使其被所有 TargetController 實例共享
        public ObjectManager objectManager; // 添加對 ObjectManager 的引用
        public CrawlerAgent crawlerAgent; // 添加对 CrawlerAgent 的引用

        void Start()
        {
            // 初始化目標位置
            targetPositions = new Vector3[3];
            targetPositions[0] = new Vector3(-32, 7, -68);
            targetPositions[1] = new Vector3(-20, -7, -68);
            targetPositions[2] = new Vector3(-8, -7, -68);

            num = 0;
        }

        /// <summary>
        /// 將目標移動到指定半徑內的隨機位置。
        /// </summary>
        public void MoveTargetToHome()
        {
            //var newTargetPos = m_startingPos + (Random.insideUnitSphere * spawnRadius);
            var newTargetPos = targetPositions[num];
            newTargetPos.y = m_startingPos.y;
            transform.position = newTargetPos;

            // 设置固定的旋转
            transform.rotation = Quaternion.Euler(-10, 0, 0);
        }

        private bool isCoolingDown = false;
        private float cooldownDuration = 3f; // 冷卻時間

        private void OnCollisionEnter(Collision col)
        {
            if (isCoolingDown) return; // 如果正在冷卻中，退出方法

            if (col.gameObject.CompareTag(tagToDetect))
            {
            onCollisionEnterEvent.Invoke(col);

            if (respawnIfTouched)
            {
                StartCoroutine(MoveAfterDelay(0.5f)); // 添加0.5秒延遲

                num += 1;

                if (num == 1)
                {
                StartCoroutine(ReloadSceneAfterDelay(2f));
                num = 0;
                }

                // 開始冷卻時間
                StartCoroutine(CooldownCoroutine());
            }
            }
        }

        private IEnumerator MoveAfterDelay(float delay)
        {
            yield return new WaitForSeconds(delay);
            MoveTargetToHome();
        }

        private IEnumerator CooldownCoroutine()
        {
            isCoolingDown = true; // 開始冷卻
            yield return new WaitForSeconds(cooldownDuration); // 等待冷卻時間結束
            isCoolingDown = false; // 冷卻結束
        }

        private IEnumerator ReloadSceneAfterDelay(float delay)
        {
            yield return new WaitForSeconds(delay);

            // 调试信息，检查 objectManager 是否为空
            if (objectManager == null)
            {
                UnityEngine.Debug.LogError("objectManager is null!");
            }
            else
            {
                objectManager.InitializeObjects();
            }

            // 调试信息，检查 crawlerAgent 是否为空
            if (crawlerAgent == null)
            {
                UnityEngine.Debug.LogError("crawlerAgent is null!");
            }
            else
            {
                // 调用 CrawlerAgent 的 OnEpisodeBegin 方法
                crawlerAgent.OnEpisodeBegin();
            }
        }


        private void OnCollisionStay(Collision col)
        {
            if (col.transform.CompareTag(tagToDetect))
            {
                onCollisionStayEvent.Invoke(col);
            }
        }

        private void OnCollisionExit(Collision col)
        {
            if (col.transform.CompareTag(tagToDetect))
            {
                onCollisionExitEvent.Invoke(col);
            }
        }

        private void OnTriggerEnter(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                onTriggerEnterEvent.Invoke(col);
            }
        }

        private void OnTriggerStay(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                onTriggerStayEvent.Invoke(col);
            }
        }

        private void OnTriggerExit(Collider col)
        {
            if (col.CompareTag(tagToDetect))
            {
                onTriggerExitEvent.Invoke(col);
            }
        }
    }
}
