using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public interface ITrain
{
    public bool WasPreTrained { get; }
    public void Train(ActionDatabase database);
    public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict);
    public (string, string[], double) Predict(string input, int top = 5);
    public void LoadModel(ActionDatabase database);
}

public class InputConsumer : MonoBehaviour
{
    [SerializeField] private TMP_InputField input;
    [Header("JSON")] public TextAsset actionJson;
    [Header("CSV")] public TextAsset actionCsv_Train;
    public TextAsset actionCsv_Test;
    [SerializeField] private UsedModel usedModel;
    public event Action<string> OnPredict;
    private const int seed = 42;
    //private StatisticalMatcher matcher;
    //[SerializeField] private ActionMatcherTester tester;

    private void Awake()
    {
        usedModel.OnModelLoad += InputData;
    }

    private void InputData()
    {
        if (actionCsv_Test != null)
        {
            //TestMod
            return;
        }

        if (actionCsv_Train != null)
        {
            //auto-fit mod
            List<TrainingSample> samples = CSVReader.ReadCSV(actionCsv_Train);
            (List<TrainingSample> trainSamples, List<TrainingSample> validationSamples) = samples.Split(seed);
            List<string> ids = samples.FillActions();
            Dictionary<string, int> dict = ids.ToDictionary();
            usedModel.Model.AutoFit(trainSamples, validationSamples, ids, dict);
            return;
        }

        ActionDatabase database = JsonConvert.DeserializeObject<ActionDatabase>(actionJson.text);

        if (usedModel.Model.WasPreTrained)
        {
            usedModel.Model.LoadModel(database);
            return;
        }

        usedModel.Model.Train(database);
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

        (string action, string[] logs, double time) = usedModel.Model.Predict(input, topK);

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