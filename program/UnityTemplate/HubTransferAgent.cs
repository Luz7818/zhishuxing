using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class HubTransferAgent : Agent
{
    [Header("Scene References")]
    public Transform selfTransform;
    public Transform targetGate;
    public List<Transform> nearbyCrowd;

    [Header("Motion")]
    public float maxSpeed = 3.0f;
    public float turnSpeed = 180f;

    [Header("Reward Weights")]
    public float stepPenalty = -0.001f;
    public float progressReward = 0.01f;
    public float congestionPenalty = -0.02f;
    public float collisionPenalty = -0.2f;
    public float successReward = 2.0f;

    [Header("Episode")]
    public float maxEpisodeSeconds = 120f;

    private Rigidbody _rb;
    private Vector3 _lastPosition;
    private float _episodeTimer;

    public override void Initialize()
    {
        _rb = GetComponent<Rigidbody>();
        if (selfTransform == null)
        {
            selfTransform = transform;
        }
    }

    public override void OnEpisodeBegin()
    {
        _episodeTimer = 0f;
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;

        selfTransform.localPosition = new Vector3(
            Random.Range(-8f, 8f),
            selfTransform.localPosition.y,
            Random.Range(-8f, 8f)
        );

        _lastPosition = selfTransform.position;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 toTarget = targetGate.position - selfTransform.position;
        sensor.AddObservation(toTarget.x);
        sensor.AddObservation(toTarget.z);

        sensor.AddObservation(_rb.linearVelocity.x);
        sensor.AddObservation(_rb.linearVelocity.z);

        foreach (Transform crowd in nearbyCrowd)
        {
            Vector3 delta = crowd.position - selfTransform.position;
            sensor.AddObservation(delta.x);
            sensor.AddObservation(delta.z);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float forward = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float turn = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector3 move = selfTransform.forward * forward * maxSpeed;
        _rb.linearVelocity = new Vector3(move.x, _rb.linearVelocity.y, move.z);
        selfTransform.Rotate(Vector3.up, turn * turnSpeed * Time.fixedDeltaTime);

        AddReward(stepPenalty);

        float lastDist = Vector3.Distance(_lastPosition, targetGate.position);
        float nowDist = Vector3.Distance(selfTransform.position, targetGate.position);
        AddReward((lastDist - nowDist) * progressReward);
        _lastPosition = selfTransform.position;

        float congestion = EstimateCongestion();
        AddReward(congestionPenalty * congestion);

        _episodeTimer += Time.fixedDeltaTime;
        if (_episodeTimer >= maxEpisodeSeconds)
        {
            EndEpisode();
        }

        if (nowDist < 1.2f)
        {
            AddReward(successReward);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxis("Vertical");
        continuousActions[1] = Input.GetAxis("Horizontal");
    }

    private float EstimateCongestion()
    {
        float congestion = 0f;
        foreach (Transform crowd in nearbyCrowd)
        {
            float d = Vector3.Distance(selfTransform.position, crowd.position);
            if (d < 2f)
            {
                congestion += (2f - d) / 2f;
            }
        }
        return congestion;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Obstacle") || collision.collider.CompareTag("Agent"))
        {
            AddReward(collisionPenalty);
        }
    }
}
