using Newtonsoft.Json;
using System.Collections.Generic;
using UnityEngine;

public class ActionMatcherTester : MonoBehaviour
{
    [Header("JSON")]
    public TextAsset actionJson;
    [Header("JSON")]
    public TextAsset preferJson;

    [Header("Test")]
    [TextArea(2, 5)]
    public string currentInput;

    [Header("Test")]
    [TextArea(2, 5)]
    public string prevAction;

    [Header("Top K")]
    public int topK = 5;

    private StatisticalMatcher matcher;

    private void Start()
    {
        ActionDatabase database =
        JsonUtility.FromJson<ActionDatabase>(
            actionJson.text
        );

        matcher = new StatisticalMatcher();
        matcher.Build(database);

        //Test(currentInput);
    }

    [ContextMenu("Test Current Input")]
    public void TestCurrent()
    {
        Test(currentInput);
    }

    public void Test(string input)
    {
        if (matcher == null)
        {
            Debug.LogWarning(
                "Matcher가 아직 초기화되지 않았습니다."
            );

            return;
        }

        List<MatchResult> results =
            matcher.Match(
                input,
                prevAction,
                topK:topK
            );

        MatchState state =
            matcher.Classify(results);

        Debug.Log(
            $"INPUT: {input}\n" +
            $"STATE: {state}"
        );

        for (int i = 0;
             i < results.Count;
             i++)
        {
            MatchResult r = results[i];

            Debug.Log(r);
        }

        if (results.Count > 0)
        {
            prevAction = results[0].ActionId;
        }
    }
}
