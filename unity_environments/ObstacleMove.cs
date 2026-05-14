using UnityEngine;

public class ObstacleMove : MonoBehaviour
{
    public float distance = 0.5f;
    public float speed = 0.5f;

    Vector3 startPos;

    void Start()
    {
        startPos = transform.position;
    }

    void Update()
    {
        float z = Mathf.Sin(Time.time * speed) * distance;
        
        transform.position = startPos + new Vector3(0f, 0f, z);
    }
}
