using System;
using UnityEngine;

public class FallingObject : MonoBehaviour
{

    void Start()
    {
        // 添加 Rigidbody 組件，使物件受到物理引擎的影響
        Rigidbody rb = gameObject.AddComponent<Rigidbody>();
        rb.useGravity = true; // 啟用重力

        // 添加 Collider 組件，使物件成為實體可碰撞
        BoxCollider boxCollider = gameObject.AddComponent<BoxCollider>();
        boxCollider.center = new Vector3(0f, 0.7f, 0.05f);
        boxCollider.size = new Vector3(1.4f, 1.4f, 0.2f);
    }
}

