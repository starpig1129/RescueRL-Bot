using UnityEngine;
using System.Collections.Generic;
using System.Diagnostics;
using System;

public class ObjectManager : MonoBehaviour
{
    public Transform crawler; // Crawler 的 Transform
    public GameObject[] objects; // 你的三個物件

    private List<Vector3> positions = new List<Vector3>(); // 十個座標位置
    private HashSet<int> chosenIndices = new HashSet<int>(); // 已選擇的位置索引


    void Start()
    {
        //UnityEngine.Debug.Log("ObjectManager Start called");
        InitializeObjects();
    }

    public void InitializeObjects()
    {
        objects = GameObject.FindGameObjectsWithTag("Target");

        AddPositions();
        SetRandomObjectPositions();
    }

    public void AddPositions()
    {
        positions.Clear();
        positions.Add(new Vector3(37, 7, -3));
        positions.Add(new Vector3(28, 7, 34));
        positions.Add(new Vector3(3, 7, 33));
        positions.Add(new Vector3(-28, 7, -5));
        positions.Add(new Vector3(-24, 7, 35));
        positions.Add(new Vector3(-29, 7, 16));
        positions.Add(new Vector3(-31, 7, -35));
        positions.Add(new Vector3(6, 7, -30));
        positions.Add(new Vector3(32, 7, -32));
        positions.Add(new Vector3(18, 7, -9));

        // 確保座標位置數量大於等於三個物件數量
        if (positions.Count < objects.Length)
        {
            UnityEngine.Debug.LogError("Not enough positions for objects!");
        }
        //UnityEngine.Debug.Log("定位完成!");
    }

    public void SetRandomObjectPositions()
    {
        // 清空已選擇的位置索引
        chosenIndices.Clear();

        // 隨機選擇不同的位置並將其分配給每個物件
        for (int i = 0; i < objects.Length; i++)
        {
            int randomIndex = UnityEngine.Random.Range(0, positions.Count);
            while (chosenIndices.Contains(randomIndex))
            {
                randomIndex = UnityEngine.Random.Range(0, positions.Count);
            }
            chosenIndices.Add(randomIndex);

            objects[i].transform.position = positions[randomIndex];
        }
        //UnityEngine.Debug.Log("移動完成!");
    }
}




