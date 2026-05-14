using UnityEngine;
using Unity.InferenceEngine;
using System.Threading.Tasks;

public class RunModel : MonoBehaviour
{
    public ModelAsset modelAsset;
    public Transform ball;
    public Transform car;
    
    private float stiffness = 20f;
    private float damping = 10f;
    private float forceLimit = 40f;
    private ArticulationBody aBody;
    private ArticulationBody[] joints;
    private float[] startJointPositions;
    private float[] lowerLimits;
    private float[] upperLimits;
    private Vector3 startPos;
    private Quaternion startRot;
    private float Tanh(float x) => (float)System.Math.Tanh(x);
    private Vector3 lastPosition;
    private float[] lastActions;

    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;
    private Tensor<float> cpuCopyTensor;
    private bool inferencing = false;
    
    private int decisionCounter = 0;
    private int decisionPeriod = 5;
    
    private Vector3 carStartPos;
    private Quaternion carStartRot;

    private void Start()
    {
        aBody = GetComponent<ArticulationBody>();
        joints = GetComponentsInChildren<ArticulationBody>();
        joints = System.Array.FindAll(joints, j => j != aBody);
        startJointPositions = new float[joints.Length];
        lowerLimits = new float[joints.Length];
        upperLimits = new float[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            var joint = joints[i];
            var drive = joint.xDrive;

            startJointPositions[i] = joints[i].jointPosition[0] * Mathf.Rad2Deg;

            lowerLimits[i] = drive.lowerLimit;
            upperLimits[i] = drive.upperLimit;

            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = forceLimit;
            drive.targetVelocity = 0f;
            joint.xDrive = drive;
        }

        startPos = transform.position;
        startRot = transform.rotation;
        lastActions = new float[joints.Length];

        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.CPU);
        
        ResetAgent();
        
        carStartPos = car.position;
        carStartRot = car.rotation;
    }

    private void FixedUpdate()
    {
        if (!inferencing)
        {
            if (decisionCounter == 0)
            {
                _ = InferenceAsync();
                decisionCounter = decisionPeriod;
            }
            decisionCounter--;
        }
        
        float agentDistance = Vector3.Distance(transform.localPosition, ball.localPosition);
        float carDistance = Vector3.Distance(car.localPosition, ball.localPosition);              

        if (agentDistance < 1.0f)
        {
            ScoreManager.instance.AddAIScore();
            ResetAgent();
            ResetCar();
        }
        
        if (carDistance < 1.0f)
        {
            ScoreManager.instance.AddPlayerScore();
            ResetAgent();
            ResetCar();
        }
        
        if (transform.localPosition.y < -1.0f)
        {
            ResetAgent();
        }
        
        if (car.localPosition.y < -1.0f)
        {
            ResetCar();
        }
    }
    
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Ground"))
        {
            ResetAgent();
        }
    }

    private async Task InferenceAsync()
    {
        inferencing = true;
        
        float[] observation = new float[38];
        
        int index = 0;      
        Vector3 localVel = transform.InverseTransformDirection(aBody.linearVelocity);
        observation[index++] = Tanh(localVel.x / 2f);
        observation[index++] = Tanh(localVel.y / 2f);
        observation[index++] = Tanh(localVel.z / 2f);
        
        Vector3 localAngVel = transform.InverseTransformDirection(aBody.angularVelocity);
        observation[index++] = Tanh(localAngVel.x / 2f);
        observation[index++] = Tanh(localAngVel.y / 2f);
        observation[index++] = Tanh(localAngVel.z / 2f);
        
        Vector3 upVector = transform.InverseTransformDirection(Vector3.up);
        observation[index++] = upVector.x;
        observation[index++] = upVector.y;
        observation[index++] = upVector.z;

        for (int i = 0; i < joints.Length; i++)
        {
            float currentPos = joints[i].jointPosition[0] * Mathf.Rad2Deg;
            float normalizedPos = Mathf.InverseLerp(lowerLimits[i], upperLimits[i], currentPos) * 2f - 1f;
            observation[index++] = normalizedPos;
            
            float currentVel = joints[i].jointVelocity[0];
            observation[index++] = Tanh(currentVel / 2f);
        }
        
        for (int i = 0; i < lastActions.Length; i++)
        {
            observation[index++] = lastActions[i];
        }
        
        Vector3 toBall = ball.position - transform.position;

        Vector3 dirToBall = toBall.normalized;
        observation[index++] = dirToBall.x;
        observation[index++] = dirToBall.y;
        observation[index++] = dirToBall.z;
        
        float distToBall = toBall.magnitude;
        observation[index++] = distToBall / 40f;
        
        float laneOffset = transform.localPosition.x - (-23.51f);
        observation[index++] = laneOffset / 2.5f;

        TensorShape shape = new TensorShape(1, 38);
        inputTensor?.Dispose();
        inputTensor = new Tensor<float>(shape, observation);
        worker.Schedule(inputTensor);
        using (var outputTensor = worker.PeekOutput() as Tensor<float>)
        {
            cpuCopyTensor = await outputTensor.ReadbackAndCloneAsync();
        }
        
        for (int i = 0; i < joints.Length; i++)
        {
            float action = cpuCopyTensor[i];
            lastActions[i] = action;
            float targetAngle = Mathf.Lerp(lowerLimits[i], upperLimits[i], (action + 1f) / 2f);
            
            var drive = joints[i].xDrive;
            drive.target = targetAngle;
            joints[i].xDrive = drive;
        }
        
        cpuCopyTensor?.Dispose();
        inferencing = false;
    }

    private void OnDestroy()
    {
        inputTensor?.Dispose();
        worker?.Dispose();
    }
    
    public void ResetAgent()
    {
        aBody.TeleportRoot(startPos, startRot);
        aBody.linearVelocity = Vector3.zero;
        aBody.angularVelocity = Vector3.zero;

        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].jointPosition = new ArticulationReducedSpace(startJointPositions[i] * Mathf.Deg2Rad);
            joints[i].jointVelocity = new ArticulationReducedSpace(0f);

            var drive = joints[i].xDrive;
            drive.target = startJointPositions[i];
            drive.targetVelocity = 0f;
            joints[i].xDrive = drive;
        }

        Physics.SyncTransforms();
        lastPosition = transform.localPosition;
        System.Array.Clear(lastActions, 0, lastActions.Length);
        decisionCounter = 0;
    }
    
    private void ResetCar()
    {
        car.position = carStartPos;
        car.rotation = carStartRot;

        Rigidbody rb = car.GetComponent<Rigidbody>();
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
    }
}
