using NaturalLanguageActionClassifier;
using System;
using System.Collections.Generic;
using System.Linq;

public static class SplitSamples
{
    private const float validationRatio = 0.2f;
    public static List<string> FillActions(this List<TrainingSample> samples)
    {
        List<string> result = samples
        .Select(s => s.Action)
        .Where(a => a != Normalizer.NULL_ID)
        .Distinct()
        .OrderBy(a => a).
        ToList();

        if (result.Count == 0)
            throw new InvalidOperationException("학습 가능한 Action이 없습니다.");

        return result;
    }

    public static Dictionary<string, int> ToDictionary(this List<string> actions)
        {
        var result = new Dictionary<string, int>();

            for (int i = 0; i<actions.Count; i++)
            result[actions[i]] = i;

        return result;
        }

    public static List<TrainingSample> Sampling(this ActionDatabase database)
    {
        List<TrainingSample> samples = new();
        foreach (var sample in database.actions)
        {
            if (sample.action_id != Normalizer.NULL_ID)
            {
                foreach (string text in sample.positive)
                {
                    samples.Add(new TrainingSample(text, sample.action_id));
                }
            }
        }
        return samples;
    }

    public static (List<TrainingSample>, List<TrainingSample>) Split(this List<TrainingSample> samples, int seed)
    {
        // Action별로 Train / Validation 분리
        var random = new Random(seed);
        Dictionary<string, int> labelToIndex = samples.FillActions().ToDictionary();

        var trainSamples = new List<TrainingSample>();
        var validationSamples = new List<TrainingSample>();

        foreach (var group in samples.Where(s => labelToIndex.ContainsKey(s.Action)).GroupBy(s => s.Action))
        {
            var shuffled = group.OrderBy(_ => random.Next()).ToList();

            int validationCount = (int)Math.Round(shuffled.Count * validationRatio);

            // 각 Action의 Train 데이터는 최소 1개 유지
            validationCount = Math.Min(
                validationCount,
                Math.Max(0, shuffled.Count - 1));

            validationSamples.AddRange(
                shuffled.Take(validationCount));

            trainSamples.AddRange(
                shuffled.Skip(validationCount));
        }

        return (trainSamples, validationSamples);
    }
}
