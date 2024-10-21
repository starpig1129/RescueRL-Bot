using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Runtime.Serialization;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System;
[RequireComponent(typeof(JointDriveController))] // 要求設置關節驅動力
public class CrawlerAgent : Agent
{

    [Header("步行速度")]
    [Range(0.1f, m_maxWalkingSpeed)]
    [SerializeField]
    [Tooltip(
        "代理將嘗試匹配的速度。\n\n" +
        "訓練:\n" +
        "對於VariableSpeed環境，此值將在每個訓練回合開始時隨機化。\n" +
        "否則代理將嘗試匹配這裡設置的速度。\n\n" +
        "推理:\n" +
        "在推理期間，VariableSpeed代理將根據此值修改其行為，" +
        "而CrawlerDynamic & CrawlerStatic代理將以訓練期間指定的速度運行"
    )]
    // 嘗試達到的步行速度
    private float m_TargetWalkingSpeed = m_maxWalkingSpeed;

    const float m_maxWalkingSpeed = 15; // 最大步行速度

    // 當前目標步行速度。因為速度為零會造成NaN，所以需要限制
    public float TargetWalkingSpeed
    {
        get { return m_TargetWalkingSpeed; }
        set { m_TargetWalkingSpeed = Mathf.Clamp(value, .1f, m_maxWalkingSpeed); }
    }
    private Socket resetSocket;
    private byte[] resetSignal;
    // 訓練期間代理將走向的目標方向。
    [Header("走向目標")]
    public Transform TargetPrefab; // 在Dynamic環境中使用的目標預製件
    private Transform m_Target; // 訓練期間代理將走向的目標。

    [Header("身體部位")][Space(10)] public Transform body;
    public Transform leg0Upper;
    public Transform leg0Lower;
    public Transform leg1Upper;
    public Transform leg1Lower;
    public Transform leg2Upper;
    public Transform leg2Lower;
    public Transform leg3Upper;
    public Transform leg3Lower;

    // 這將用作觀察的穩定模型空間參考點
    // 因為訓練期間ragdoll可能會不規則移動，使用穩定的參考變換可以改善學習
    OrientationCubeController m_OrientationCube;

    // 指向目標的指示器圖形遊戲對象
    DirectionIndicator m_DirectionIndicator;
    JointDriveController m_JdController;

    [Header("腳部著地視覺化")]
    [Space(10)]
    public bool useFootGroundedVisualization;

    public MeshRenderer foot0;
    public MeshRenderer foot1;
    public MeshRenderer foot2;
    public MeshRenderer foot3;
    public Material groundedMaterial;
    public Material unGroundedMaterial;

    public override void Initialize()
    {
        SpawnTarget(TargetPrefab, transform.position);

        m_OrientationCube = GetComponentInChildren<OrientationCubeController>();
        m_DirectionIndicator = GetComponentInChildren<DirectionIndicator>();
        m_JdController = GetComponent<JointDriveController>();

        // 設置每個身體部位
        m_JdController.SetupBodyPart(body);
        m_JdController.SetupBodyPart(leg0Upper);
        m_JdController.SetupBodyPart(leg0Lower);
        m_JdController.SetupBodyPart(leg1Upper);
        m_JdController.SetupBodyPart(leg1Lower);
        m_JdController.SetupBodyPart(leg2Upper);
        m_JdController.SetupBodyPart(leg2Lower);
        m_JdController.SetupBodyPart(leg3Upper);
        m_JdController.SetupBodyPart(leg3Lower);

        resetSignal = BitConverter.GetBytes(1);
    }
    private void SendResetSignal()
    {
        try
        {
            using (Socket resetSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                resetSocket.Connect(new IPEndPoint(IPAddress.Parse("127.0.0.1"), 7000));
                resetSocket.Send(resetSignal);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending reset signal: {e.Message}");
        }
    }
    /// <summary>
    /// 在指定位置生成一個目標預製件
    /// </summary>
    /// <param name="prefab"></param>
    /// <param name="pos"></param>
    void SpawnTarget(Transform prefab, Vector3 pos)
    {
        m_Target = Instantiate(prefab, pos, Quaternion.identity, transform.parent);
    }

    public ObjectManager objectManager; // 添加對 ObjectManager 的引用
    public TargetController targetController;
    /// <summary>
    /// 
    /// 循環遍歷身體部位並將它們重置到初始條件。
    /// </summary>
    public override void OnEpisodeBegin()
    {
        foreach (var bodyPart in m_JdController.bodyPartsDict.Values)
        {
            bodyPart.Reset(bodyPart);
        }
        // 调试信息，检查 objectManager 是否为空
        if (objectManager == null)
        {
            UnityEngine.Debug.LogError("objectManager is null!");
        }
        else
        {
            TargetController.num = 0;
            objectManager.InitializeObjects();
        }
        // 隨機開始旋轉以幫助泛化
        body.rotation = Quaternion.Euler(0, Random.Range(0.0f, 360.0f), 0);

        UpdateOrientationObjects();

        // 設定我們的目標步行速度
        TargetWalkingSpeed = Random.Range(0.1f, m_maxWalkingSpeed);

        // 發送重製訊號到Python
        SendResetSignal();
    }

    /// <summary>
    /// 添加有關每個身體部位的相關信息到觀察中。
    /// </summary>
    public void CollectObservationBodyPart(BodyPart bp, VectorSensor sensor)
    {
        // 地面檢查
        sensor.AddObservation(bp.groundContact.touchingGround); // 此部位是否接觸地面

        if (bp.rb.transform != body)
        {
            sensor.AddObservation(bp.currentStrength / m_JdController.maxJointForceLimit);
        }
    }

    /// <summary>
    /// 循環遍歷身體部位並將它們添加到觀察中。
    /// </summary>
    public override void CollectObservations(VectorSensor sensor)
    {
        var cubeForward = m_OrientationCube.transform.forward;

        // 我們想要匹配的速度
        var velGoal = cubeForward * TargetWalkingSpeed;
        // ragdoll的平均速度
        var avgVel = GetAvgVelocity();

        // 當前ragdoll速度，已正規化
        sensor.AddObservation(Vector3.Distance(velGoal, avgVel));
        // 平均身體速度相對於方向立方體
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(avgVel));
        // 速度目標相對於方向立方體
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformDirection(velGoal));
        // 旋轉差異
        sensor.AddObservation(Quaternion.FromToRotation(body.forward, cubeForward));

        // 添加目標位置相對於方向立方體的位置
        sensor.AddObservation(m_OrientationCube.transform.InverseTransformPoint(m_Target.transform.position));

        RaycastHit hit;
        float maxRaycastDist = 10;
        if (Physics.Raycast(body.position, Vector3.down, out hit, maxRaycastDist))
        {
            sensor.AddObservation(hit.distance / maxRaycastDist);
        }
        else
            sensor.AddObservation(1);

        foreach (var bodyPart in m_JdController.bodyPartsList)
        {
            CollectObservationBodyPart(bodyPart, sensor);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // 包含所有身體部位的字典在jdController中
        var bpDict = m_JdController.bodyPartsDict;

        var continuousActions = actionBuffers.ContinuousActions;
        var i = -1;
        // 選擇一個新的目標關節旋轉
        bpDict[leg0Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg1Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg2Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg3Upper].SetJointTargetRotation(continuousActions[++i], continuousActions[++i], 0);
        bpDict[leg0Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg1Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg2Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);
        bpDict[leg3Lower].SetJointTargetRotation(continuousActions[++i], 0, 0);

        // 更新關節力量
        bpDict[leg0Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Upper].SetJointStrength(continuousActions[++i]);
        bpDict[leg0Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg1Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg2Lower].SetJointStrength(continuousActions[++i]);
        bpDict[leg3Lower].SetJointStrength(continuousActions[++i]);
    }

    void FixedUpdate()
    {
        UpdateOrientationObjects();

        // 如果啟用，當腳接地時，腳會亮起綠色。
        // 這只是一種視覺化，並不是必須的功能
        if (useFootGroundedVisualization)
        {
            foot0.material = m_JdController.bodyPartsDict[leg0Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot1.material = m_JdController.bodyPartsDict[leg1Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot2.material = m_JdController.bodyPartsDict[leg2Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
            foot3.material = m_JdController.bodyPartsDict[leg3Lower].groundContact.touchingGround
                ? groundedMaterial
                : unGroundedMaterial;
        }

        var cubeForward = m_OrientationCube.transform.forward;

        // 根據以下元素混合設定這一步的獎勵。
        // a. 匹配目標速度
        // 如果完美匹配，此獎勵將接近1，偏差時接近0
        var matchSpeedReward = GetMatchingVelocityReward(cubeForward * TargetWalkingSpeed, GetAvgVelocity());

        // b. 旋轉對齊目標方向。
        // 如果完美面對目標方向，此獎勵將接近1，偏差時接近0
        var lookAtTargetReward = (Vector3.Dot(cubeForward, body.forward) + 1) * .5F;

        AddReward(matchSpeedReward * lookAtTargetReward);
    }

    /// <summary>
    /// 更新OrientationCube和DirectionIndicator
    /// </summary>
    void UpdateOrientationObjects()
    {
        m_OrientationCube.UpdateOrientation(body, m_Target);
        if (m_DirectionIndicator)
        {
            m_DirectionIndicator.MatchOrientation(m_OrientationCube.transform);
        }
    }

    /// <summary>
    /// 返回所有身體部位的平均速度
    /// 單獨使用身體的速度已顯示導致四肢更不穩定的移動
    /// 使用平均值有助於防止這種不穩定的移動
    /// </summary>
    Vector3 GetAvgVelocity()
    {
        Vector3 velSum = Vector3.zero;
        Vector3 avgVel = Vector3.zero;

        // 所有RBS
        int numOfRb = 0;
        foreach (var item in m_JdController.bodyPartsList)
        {
            numOfRb++;
            velSum += item.rb.velocity;
        }

        avgVel = velSum / numOfRb;
        return avgVel;
    }

    /// <summary>
    /// 實際速度與目標步行速度之間的差異的規範化值。
    /// </summary>
    public float GetMatchingVelocityReward(Vector3 velocityGoal, Vector3 actualVelocity)
    {
        // 我們的實際速度與目標速度之間的距離
        var velDeltaMagnitude = Mathf.Clamp(Vector3.Distance(actualVelocity, velocityGoal), 0, TargetWalkingSpeed);

        // 在一個從1降到0的下降曲線上返回值
        // 如果完美匹配，此獎勵將接近1，偏差時接近0
        return Mathf.Pow(1 - Mathf.Pow(velDeltaMagnitude / TargetWalkingSpeed, 2), 2);
    }

    /// <summary>
    /// 代理接觸到目標
    /// </summary>
    public void TouchedTarget()
    {
        AddReward(1f);
    }

    /*test
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("RandomObject"))
        {
            ObjectManager objectManager = FindObjectOfType<ObjectManager>();
            if (objectManager != null)
            {
                objectManager.MoveObject(other.gameObject);
            }
        }
    }
    */
}
