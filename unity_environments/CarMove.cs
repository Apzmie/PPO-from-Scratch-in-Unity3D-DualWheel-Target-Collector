using UnityEngine;
using UnityEngine.InputSystem;

public class CarMove : MonoBehaviour
{
    public Rigidbody body;

    public float acceleration = 1f;
    public float turnSpeed = 100f;
    public float maxSpeed = 1.5f;

    private float currentTurn;
    private float speed;
    
    bool up;
    bool down;
    bool left;
    bool right;

    void Start()
    {
        currentTurn = body.transform.eulerAngles.y;
    }

    void FixedUpdate()
    {
        float move = 0f;
        float turn = 0f;

        if (Keyboard.current.upArrowKey.isPressed || up) move += 1f;
        if (Keyboard.current.downArrowKey.isPressed || down) move -= 1f;
        if (Keyboard.current.leftArrowKey.isPressed || left) turn -= 1f;
        if (Keyboard.current.rightArrowKey.isPressed || right) turn += 1f;

        currentTurn += turn * turnSpeed * Time.fixedDeltaTime;
        Quaternion rot = Quaternion.Euler(0f, currentTurn, 0f);
        body.MoveRotation(rot);

        if (Mathf.Abs(move) > 0.01f)
        {
            speed += move * acceleration * Time.fixedDeltaTime;
        }
        else
        {
            speed = Mathf.Lerp(speed, 0f, 2f * Time.fixedDeltaTime);
        }

        speed = Mathf.Clamp(speed, -maxSpeed, maxSpeed);
        Vector3 velocity = body.transform.forward * speed;

        body.linearVelocity = new Vector3(
            velocity.x,
            body.linearVelocity.y,
            velocity.z
        );
    }
    
    public void UpPress()  { up = true; }
    public void UpRelease(){ up = false; }

    public void DownPress()  { down = true; }
    public void DownRelease(){ down = false; }

    public void LeftPress()  { left = true; }
    public void LeftRelease(){ left = false; }

    public void RightPress()  { right = true; }
    public void RightRelease(){ right = false; }
}
