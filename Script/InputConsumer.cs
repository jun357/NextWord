using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

[Serializable]
public struct EvalSample
{
    public string input;
    public EscapeAction expectedActionId;
}

public interface ITrain
{
    public void Train(ActionDatabase database);
    public (EscapeAction, string[], double) Predict(string input, int top = 5);
}

public interface IEval
{
    public void Eval(List<EvalSample> samples);
    public (EscapeAction, string[], double) Predict(string input, int top = 5);
}

public class InputConsumer : MonoBehaviour
{
    [SerializeField] private TMP_InputField input;
    [Header("JSON")] public TextAsset actionJson;
    [SerializeField] private UsedModel usedModel;
    public event Action<EscapeAction> OnPredict;
    //private StatisticalMatcher matcher;
    //[SerializeField] private ActionMatcherTester tester;

    private void Awake()
    {
        usedModel.OnModelLoad += InputData;
    }

    private void InputData()
    {
        usedModel.Model.Train(JsonConvert.DeserializeObject<ActionDatabase>(actionJson.text));
    }

    public void OnEndEdit()
    {
        Test(input.text);
    }

    public void Test(string input, int topK = 5)
    {
        if (usedModel == null || usedModel.Model == null)
        {
            Debug.LogWarning(
                "Matcher가 아직 초기화되지 않았습니다."
            );

            return;
        }

        (EscapeAction action, string[] logs, double time) = usedModel.Model.Predict(input, topK);

        //MatchState state = matcher.Classify(results);

        Debug.Log($"INPUT: {input} TIME: {time}");
        /*Debug.Log(
            $"INPUT: {input}\n" +
            $"STATE: {state}"
        );*/

        foreach (string log in logs)
        {
            Debug.Log(log);
        }

        OnPredict?.Invoke(action);
    }
}