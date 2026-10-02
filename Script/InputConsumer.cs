using Newtonsoft.Json;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class InputConsumer : MonoBehaviour
{
    [SerializeField] private TMP_InputField input;
    [Header("JSON")] public TextAsset actionJson;
    private StatisticalMatcher matcher;
    //[SerializeField] private ActionMatcherTester tester;

    private void Awake()
    {
        matcher = new StatisticalMatcher(
            JsonConvert.DeserializeObject<ActionDatabase>(
                actionJson.text
            )
        );
    }

    public void OnEndEdit()
    {
        //tester.
            Test(input.text, matcher);
    }

    public void Test(string input, StatisticalMatcher matcher, int topK = 5)
    {
        if (matcher == null)
        {
            Debug.LogWarning(
                "Matcher가 아직 초기화되지 않았습니다."
            );

            return;
        }

        IReadOnlyList<MatchResult> results =
            matcher.Match(
                input,
                //prevAction,
                topK: topK
            );

        MatchState state = matcher.Classify(results);

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

        /*if (results.Count > 0)
        {
            prevAction = results[0].ActionId;
        }*/
    }
}