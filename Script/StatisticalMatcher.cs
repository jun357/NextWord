using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

[Serializable]
public struct SlotVector
{
    public float intend_char;
    public float intend_word;
    public float target_char;
    public float target_word;
    public float context_char;
    public float context_word;
    public float action_char;
    public float action_word;
}

public enum SlotFeature
{
    Intend = 0,
    Target = 1,
    Context = 2,
    Action = int.MaxValue,
}

[Serializable]
public readonly struct StatisticalFeature
{
    private readonly Dictionary<SlotFeature, string> _slotFeature;
    public string SlotFeatures(SlotFeature feature) => _slotFeature.GetValueOrDefault(feature);

    public string Intend => _slotFeature?.GetValueOrDefault(SlotFeature.Intend) ?? Normalizer.NULL_LABEL;
    public string Target => _slotFeature?.GetValueOrDefault(SlotFeature.Target) ?? Normalizer.NULL_LABEL;
    public string Context => _slotFeature?.GetValueOrDefault(SlotFeature.Context) ?? Normalizer.NULL_LABEL;

    [JsonConstructor]
    public StatisticalFeature(string Intend, string Target, string Context)
    {
        _slotFeature = new()
        {
            { SlotFeature.Intend, Intend },
            { SlotFeature.Target, Target },
            { SlotFeature.Context, Context },
        };
    }
}

[Serializable]
public readonly struct StatisticalWeight
{
    public readonly float intendWeight;
    public readonly float targetWeight;
    public readonly float contextWeight;
    public readonly float characterWeight;
    public readonly float wordWeight;

    [JsonConstructor]
    public StatisticalWeight(float w1, float w2, float w3, float w4, float w5)
    {
        intendWeight = w1;
        targetWeight = w2;
        contextWeight = w3;
        characterWeight = w4;
        wordWeight = w5;
    }
}

public class StatisticalMatcher : MonoBehaviour, ITrain
{
    [Header("JSON")] public TextAsset pre_trained;
    public TextAsset label_data;
    private Dictionary<string, StatisticalFeature> feature_table;
    public IEnumerable<string> Actions => feature_table.Keys;
    [SerializeField] private UsedModel usedModel;
    [SerializeField] private WeightOptimizer optimizer;

    // =========================================================
    // Components
    // =========================================================

    private readonly List<ActionTrainingSample> samples = new();
    private readonly Dictionary<(SlotFeature, string), int> keyset = new();

    private readonly FeatureScorer featureScorer = new();

    private SlotMatcher slotMatcher;

    private readonly MatchDecision decision = new();

    // =========================================================
    // Weights
    // =========================================================

    //public float[] weights = { 1f,1f, 1f,1f, 1f,1f, 1f,1f };
    public Dictionary<string, float[]> weights;

    public bool WasPreTrained => pre_trained != null;

    private void MakeSlotMatcher()
    {
        slotMatcher = new(featureScorer);
        slotMatcher.CharacterWeight = 1f;
        slotMatcher.WordWeight = 1f;
    }

    private void Start()
    {
        feature_table = JsonConvert.DeserializeObject<Dictionary<string, StatisticalFeature>>(label_data.text);
        if (usedModel != null)
            usedModel.Model = this;
    }

    // =========================================================
    // Build
    // =========================================================

    void ITrain.Train(ActionDatabase database) => Build(database);

    private void Build(List<TrainingSample> trainSamples, Dictionary<string, StatisticalFeature> feature_table)
    {
        MakeSlotMatcher();

        Dictionary<string, ActionTrainingSample> sampleCache = new();
        keyset.Clear();

        foreach (TrainingSample trainSample in trainSamples)
        {
            string normalized = trainSample.Text.Normalize();

            var ngram = FeatureScorer.ExtractCharacterNGrams(normalized);
            var words = FeatureScorer.TokenizeWords(normalized);

            if (!sampleCache.TryGetValue(trainSample.Action, out ActionTrainingSample sample))
            {
                sample = new(trainSample.Action, feature_table.GetValueOrDefault(trainSample.Action));
                sampleCache[trainSample.Action] = sample;
            }

            sample.Populate(ngram, words);
            keyset.AddInto((SlotFeature.Intend, sample.Feature.Intend), 1);
            keyset.AddInto((SlotFeature.Target, sample.Feature.Target), 1);
            keyset.AddInto((SlotFeature.Context, sample.Feature.Context), 1);
        }

        samples.Clear();
        foreach (var sample in sampleCache.Values)
        {
            samples.Add(sample);
        }

        featureScorer.Build(samples);

        UnityEngine.Debug.Log(samples.Count);
    }

    private void Build(ActionDatabase database)
    {
        MakeSlotMatcher();

        samples.Clear();

        foreach (
            ActionDefinition definition
            in database.actions)
        {
            ActionTrainingSample sample = new(definition.action_id, new(definition.intend, definition.target, definition.context));

            foreach (string text in definition.positive)
            {
                string normalized = text.Normalize();
                FeatureScorer.PopulateTF(sample.charFRQ, FeatureScorer.ExtractCharacterNGrams(normalized));
                FeatureScorer.PopulateTF(sample.wordFRQ, FeatureScorer.TokenizeWords(normalized));
            }

            //NormalizeDictionary(sample.Positive);

            samples.Add(sample);
        }

        featureScorer.Build(samples);

        UnityEngine.Debug.Log(samples.Count);
    }

    /*public (string, string[], double) Predict(string input, int top = 5)
    {
        var stopwatch = Stopwatch.StartNew();
        Vector2[] ws = weights.ToVector2Array();
        var result = Match(input, ws[0], ws[1], ws[2], ws[3], top);
        stopwatch.Stop();
        if (result == null || result.Count == 0)
            return (Normalizer.NULL_ID, System.Array.Empty<string>(), stopwatch.Elapsed.TotalMilliseconds);
        return (result[0].ActionId, result.Select(x => x.ToString()).ToArray(), stopwatch.Elapsed.TotalMilliseconds);
    }*/

    // =========================================================
    // Match
    // =========================================================

    public (Vector2[], Dictionary<(SlotFeature,string),Vector2>) TextToScoreMap(string text)
    {
        string current = text.Normalize();

        List<string> grams = FeatureScorer.ExtractCharacterNGrams(current);
        Dictionary<string, int> charTf = new();
        FeatureScorer.PopulateTF(charTf, grams);

        string[] words = FeatureScorer.TokenizeWords(current);
        Dictionary<string, int> wordTf = new();
        FeatureScorer.PopulateTF(wordTf, words);

        return slotMatcher.Match(charTf, wordTf, samples);
    }

    public string ScoreMapToScore(Vector2[] scores, Dictionary<(SlotFeature, string), Vector2> featureScore, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4)
    {
        Dictionary<(SlotFeature, string), float> cache = new();
        foreach ((var key, int size) in keyset)
        {
            Vector2 weight = key.Item1 switch
            {
                SlotFeature.Intend => v1,
                SlotFeature.Target => v2,
                SlotFeature.Context => v3,
                _ => new Vector2(0, 0)
            };
            cache[key] = Vector2.Dot(featureScore[key], weight) / size;
        }

        int idx = 0;
        string max_id = Normalizer.NULL_ID;
        float max_score = -1;
        foreach (ActionTrainingSample sample in samples)
        {
            float intendScore = cache.GetValueOrDefault((SlotFeature.Intend, sample.Feature.Intend));
            float targetScore = cache.GetValueOrDefault((SlotFeature.Target, sample.Feature.Target));
            float contextScore = cache.GetValueOrDefault((SlotFeature.Context, sample.Feature.Context));
            float actionScore = Vector2.Dot(scores[idx], v4) / sample.sampleCount;
            float finalScore = intendScore + targetScore + contextScore + actionScore;

            if (finalScore > max_score)
            {
                max_id = sample.ActionId;
                max_score = finalScore;
            }

            idx++;
        }

        return max_id;
    }

    public Dictionary<string, float> ScoreMapToScore(Vector2[] scores, Dictionary<(SlotFeature, string), Vector2> featureScore, Dictionary<string, float[]> model)
    {
        Dictionary<string, float> result = new();

        int idx = 0;
        foreach (ActionTrainingSample sample in samples)
        {
            float[] weights = model[sample.ActionId];
            Vector2[] vectors = weights.ToVector2Array();

            var key = (SlotFeature.Intend, sample.Feature.Intend);
            int count = keyset[key];
            Vector2 score = featureScore[key];
            float intendScore = Vector2.Dot(score, vectors[0]) / count;

            key = (SlotFeature.Target, sample.Feature.Target);
            count = keyset[key];
            score = featureScore[key];
            float targetScore = Vector2.Dot(score, vectors[1]) / count;

            key = (SlotFeature.Context, sample.Feature.Context);
            count = keyset[key];
            score = featureScore[key];
            float contextScore = Vector2.Dot(score, vectors[2]) / count;

            float actionScore = Vector2.Dot(scores[idx], vectors[3]) / sample.sampleCount;

            float finalScore = intendScore + targetScore + contextScore + actionScore;

            result[sample.ActionId] = finalScore;

            idx++;
        }

        return result;
    }

    public IReadOnlyList<MatchResult> Match(string currentInput, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4, int topK = 5)
    {
        if (string.IsNullOrWhiteSpace(currentInput))
        {
            return Array.Empty<MatchResult>();
        }

        string current = currentInput.Normalize();

        List<string> grams = FeatureScorer.ExtractCharacterNGrams(current);
        Dictionary<string, int> charTf = new();
        FeatureScorer.PopulateTF(charTf, grams);
        
        string[] words = FeatureScorer.TokenizeWords(current);
        Dictionary<string, int> wordTf = new();
        FeatureScorer.PopulateTF(wordTf, words);

        (var scores, var featureScore) = slotMatcher.Match(charTf, wordTf, samples);

        List<MatchResult> results = new();
        Dictionary<(SlotFeature, string), float> cache = new();

        foreach ((var key, int size) in keyset)
        {
            Vector2 weight = key.Item1 switch
            {
                SlotFeature.Intend => v1,
                SlotFeature.Target => v2,
                SlotFeature.Context => v3,
                _ => new Vector2(0, 0)
            };
            cache[key] = Vector2.Dot(featureScore[key], weight) / size;
        }

        int idx = 0;
        foreach (ActionTrainingSample sample in samples)
        {
            float intendScore = cache.GetValueOrDefault((SlotFeature.Intend, sample.Feature.Intend));
            float targetScore = cache.GetValueOrDefault((SlotFeature.Target, sample.Feature.Target));
            float contextScore = cache.GetValueOrDefault((SlotFeature.Context, sample.Feature.Context));
            float actionScore = Vector2.Dot(scores[idx], v4) / sample.sampleCount;

            float finalScore = intendScore + targetScore + contextScore + actionScore;

            results.Add(
                new MatchResult(
                    sample.ActionId,
                    sample.Feature,
                    intendScore,
                    targetScore,
                    contextScore,
                    actionScore,
                    finalScore
                    ));

            idx++;
        }

        return results.OrderByDescending(x => x.FinalScore).Take(topK).ToList();
    }

    // =========================================================
    // Classify
    // =========================================================

    const float rate = 1 - 0.8235294117647059f;

    public MatchState Classify(
        IReadOnlyList<MatchResult> results)
    {
        return decision.Classify(results);
    }

    public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
    {
        Build(trainSamples, feature_table);
        weights = WeightOptimizer.InitialModel(Actions, mask_intend:false, mask_target:false, mask_context: false, mask_action: false);
    }

    /*public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
    {
        (List<TrainingSample> dataSet, List<TrainingSample> trainSet) = trainSamples.Split(42, rate);
        Build(dataSet, feature_table);
        optimizer.OptimizeWeightsHacker(trainSet, validationSamples);
        Build(trainSamples, feature_table);
    }*/

    public void LoadModel(ActionDatabase database)
    {
        Build(database);
    }

    public float TrainOne(
        Vector2[] scores,
        Dictionary<(SlotFeature, string), Vector2> featureScore,
        Dictionary<string, float[]> model,
        string targetActionId,
        float learningRate)
    {
        int actionCount = samples.Count;
        if (actionCount == 0)
            return 0f;

        // 1. Forward: 각 액션 점수 계산
        var logits = ScoreMapToScore(scores, featureScore, model);
        var probabilities = Softmax(logits);

        // 2. Cross Entropy loss
        float loss = CalculateLoss(probabilities, targetActionId);

        // 3. Backpropagation + SGD
        for (int i = 0; i < actionCount; i++)
        {
            ActionTrainingSample sample = samples[i];

            float error = probabilities[sample.ActionId] - (sample.ActionId == targetActionId ? 1f : 0f);

            float[] weights = model[sample.ActionId];

            var key = (SlotFeature.Intend, sample.Feature.Intend);
            Vector2 intend = featureScore[key];
            int intendCount = keyset[key];

            key = (SlotFeature.Target, sample.Feature.Target);
            Vector2 target = featureScore[key];
            int targetCount = keyset[key];

            key = (SlotFeature.Context, sample.Feature.Context);
            Vector2 context = featureScore[key];
            int contextCount = keyset[key];

            float actionSampleCount = sample.sampleCount;

            Vector2[] inputs =
            {
                intend / intendCount,
                target / targetCount,
                context / contextCount,
                scores[i] / actionSampleCount
            };

            // 각 Vector2 가중치의 두 성분을 업데이트
            for (int j = 0; j < 4; j++)
            {
                int k = j * 2;

                weights[k] -= learningRate * error * inputs[j].x;
                weights[k + 1] -= learningRate * error * inputs[j].y;
            }
        }

        return loss;
    }

    private Dictionary<string, float> Softmax(
    Dictionary<string, float> logits)
    {
        var probabilities = new Dictionary<string, float>();

        if (logits.Count == 0)
            return probabilities;

        float maxLogit = logits.Values.Max();
        float sumExp = 0f;

        foreach (var (actionId, logit) in logits)
        {
            float exp = MathF.Exp(logit - maxLogit);
            probabilities[actionId] = exp;
            sumExp += exp;
        }

        foreach (string actionId in probabilities.Keys.ToArray())
            probabilities[actionId] /= sumExp;

        return probabilities;
    }

    private float CalculateLoss(
    Dictionary<string, float> probabilities,
    string targetActionId)
    {
        if (!probabilities.TryGetValue(targetActionId, out float probability))
            throw new ArgumentException(
                "Target action is not present in candidates.",
                nameof(targetActionId));

        return -MathF.Log(MathF.Max(probability, 1e-7f));
    }

    public (float loss, bool correct) EvaluateOne(
    Vector2[] scores,
    Dictionary<(SlotFeature, string), Vector2> featureScore,
    Dictionary<string, float[]> model,
    string targetActionId)
    {
        var logits = ScoreMapToScore(scores, featureScore, model);
        var probabilities = Softmax(logits);

        float loss = CalculateLoss(probabilities, targetActionId);

        string maxKey = null;
        float maxValue = float.MinValue;

        foreach (var item in probabilities)
        {
            if (item.Value > maxValue)
            {
                maxValue = item.Value;
                maxKey = item.Key;
            }
        }

        return (loss, maxKey == targetActionId);
    }

    /*public TestResult Test(List<TrainingSample> testSamples)
    {
        TestResult result = new();

        if (testSamples == null || testSamples.Count == 0)
            return result;

        double totalLoss = 0;

        foreach (TrainingSample sample in testSamples)
        {
            // 실제 정답
            string targetActionId = sample.Action;

            // 예측
            var (predictedId, _, _) = Predict(sample.Text);

            result.TotalSamples++;

            if (predictedId == targetActionId)
            {
                result.CorrectPredictions++;
            }
            else
            {
                result.IncorrectPredictions++;
            }

            // 정답 액션의 점수로 Cross Entropy를 계산할 수 있도록
            // 전체 액션의 점수를 가져온다.
            var matches = Match(
                sample.Text,
                weights.ToVector2Array()[0],
                weights.ToVector2Array()[1],
                weights.ToVector2Array()[2],
                weights.ToVector2Array()[3],
                samples.Count);

            if (matches.Count > 0)
            {
                float maxScore = matches.Max(x => x.FinalScore);

                float sumExp = 0;
                foreach (var match in matches)
                    sumExp += MathF.Exp(match.FinalScore - maxScore);

                var targetMatch = matches.FirstOrDefault(
                    x => x.ActionId == targetActionId);

                if (targetMatch != null)
                {
                    double logSumExp =
                        maxScore + Math.Log(sumExp);

                    totalLoss += logSumExp - targetMatch.FinalScore;
                }
                else
                {
                    // 정답 액션이 모델 후보에 없는 경우
                    totalLoss += -Math.Log(1e-7);
                }
            }
        }

        result.Accuracy =
            result.TotalSamples > 0
                ? (double)result.CorrectPredictions / result.TotalSamples
                : 0;

        result.AverageLoss =
            result.TotalSamples > 0
                ? totalLoss / result.TotalSamples
                : 0;

        return result;
    }*/

    private Dictionary<string, float> CalculateScores(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return new Dictionary<string, float>();

        if (weights == null || weights.Count == 0)
            throw new InvalidOperationException(
                "Model weights have not been initialized.");

        string normalized = input.Normalize();

        var grams = FeatureScorer.ExtractCharacterNGrams(normalized);
        Dictionary<string, int> charTf = new();
        FeatureScorer.PopulateTF(charTf, grams);

        var words = FeatureScorer.TokenizeWords(normalized);
        Dictionary<string, int> wordTf = new();
        FeatureScorer.PopulateTF(wordTf, words);

        var (scores, featureScore) = slotMatcher.Match(charTf, wordTf, samples);

        return ScoreMapToScore(scores, featureScore, weights);
    }

    public (string, string[], double) Predict(
    string input,
    int top = 5)
    {
        var stopwatch = Stopwatch.StartNew();

        var scores = CalculateScores(input);

        stopwatch.Stop();

        if (scores.Count == 0)
        {
            return (
                Normalizer.NULL_ID,
                Array.Empty<string>(),
                stopwatch.Elapsed.TotalMilliseconds);
        }

        var ranked = scores
            .OrderByDescending(x => x.Value)
            .Take(Math.Max(1, top))
            .Select(x => new MatchResult(
                x.Key,
                samples.First(s => s.ActionId == x.Key).Feature,
                0f, 0f, 0f, 0f,
                x.Value))
            .ToList();

        return (
            ranked[0].ActionId,
            ranked.Select(x => x.ToString()).ToArray(),
            stopwatch.Elapsed.TotalMilliseconds);
    }

    public TestResult Test(List<TrainingSample> testSamples)
    {
        TestResult result = new();

        if (testSamples == null || testSamples.Count == 0)
            return result;

        double totalLoss = 0;

        foreach (TrainingSample sample in testSamples)
        {
            Dictionary<string, float> logits = CalculateScores(sample.Text);

            if (logits.Count == 0)
                continue;

            result.TotalSamples++;

            string predictedId = logits.OrderByDescending(x => x.Value).First().Key;

            if (predictedId == sample.Action)
                result.CorrectPredictions++;
            else
                result.IncorrectPredictions++;

            // 전체 액션 점수로 안정적인 Cross Entropy 계산
            float maxLogit = logits.Values.Max();
            double sumExp = 0;

            foreach (float logit in logits.Values)
                sumExp += Math.Exp(logit - maxLogit);

            if (logits.TryGetValue(sample.Action, out float targetLogit))
            {
                totalLoss +=
                    Math.Log(sumExp) + maxLogit - targetLogit;
            }
            else
            {
                totalLoss += -Math.Log(1e-7);
            }
        }

        if (result.TotalSamples > 0)
        {
            result.Accuracy = (double)result.CorrectPredictions / result.TotalSamples;

            result.AverageLoss = totalLoss / result.TotalSamples;
        }

        return result;
    }

    /*public void Test2(string currentInput, string trueLabel)
    {
        string current = currentInput.Normalize();

        List<string> grams = FeatureScorer.ExtractCharacterNGrams(current);
        Dictionary<string, int> charTf = new();
        FeatureScorer.PopulateTF(charTf, grams);

        string[] words = FeatureScorer.TokenizeWords(current);
        Dictionary<string, int> wordTf = new();
        FeatureScorer.PopulateTF(wordTf, words);

        var featureScore = slotMatcher.Match2(charTf, wordTf, samples);

        var result = slotMatcher.GetBestMatch(featureScore);

        string bestIntend = result[SlotFeature.Intend].Item1;
        string bestTarget = result[SlotFeature.Target].Item1;
        string bestContext = result[SlotFeature.Context].Item1;
        string bestAction = result[SlotFeature.Action].Item1;

        string trueIntend = feature_table.GetValueOrDefault(trueLabel).Intend;
        string trueTarget = feature_table.GetValueOrDefault(trueLabel).Target;
        string trueContext = feature_table.GetValueOrDefault(trueLabel).Context;
        string trueAction = trueLabel;

        bool intendCorrect = bestIntend == trueIntend;
        bool targetCorrect = bestTarget == trueTarget;
        bool contextCorrect = bestContext == trueContext;
        bool actionCorrect = bestAction == trueAction;
    }*/

    public EvaluationResult Test2(List<TrainingSample> testSamples)
    {
        EvaluationResult eval = new();

        foreach (var sample in testSamples)
        {
            string current = sample.Text.Normalize();

            List<string> grams = FeatureScorer.ExtractCharacterNGrams(current);

            Dictionary<string, int> charTf = new();
            FeatureScorer.PopulateTF(charTf, grams);

            string[] words = FeatureScorer.TokenizeWords(current);

            Dictionary<string, int> wordTf = new();
            FeatureScorer.PopulateTF(wordTf, words);

            var featureScore = slotMatcher.Match2(charTf, wordTf, samples);

            var result = slotMatcher.GetBestMatch(featureScore);

            if (!feature_table.TryGetValue(sample.Action, out var truth))
                continue;

            // 실제 정답
            string trueIntend = truth.Intend;
            string trueTarget = truth.Target;
            string trueContext = truth.Context;
            string trueAction = sample.Action;

            // 예측 결과
            string bestIntend = result[SlotFeature.Intend].Item1;
            string bestTarget = result[SlotFeature.Target].Item1;
            string bestContext = result[SlotFeature.Context].Item1;
            string bestAction = result[SlotFeature.Action].Item1;

            // 평가
            eval.Add(SlotFeature.Intend, trueIntend, bestIntend);
            eval.Add(SlotFeature.Target, trueTarget, bestTarget);
            eval.Add(SlotFeature.Context, trueContext, bestContext);
            eval.Add(SlotFeature.Action, trueAction, bestAction);
        }

        return eval;
    }

    public void PrintTopConfusions(
        EvaluationResult eval,
        SlotFeature feature)
    {
        var matrix = eval.ConfusionMatrices[feature];

        var confusions = matrix
            .SelectMany(actual => actual.Value
                .Where(predicted => actual.Key != predicted.Key)
                .Select(predicted => new
                {
                    Actual = actual.Key,
                    Predicted = predicted.Key,
                    Count = predicted.Value
                }))
            .OrderByDescending(x => x.Count)
            .ThenBy(x => x.Actual)
            .Take(10)
            .ToList();

        UnityEngine.Debug.Log($"\n===== {feature} Top Confusions =====");

        if (confusions.Count == 0)
        {
            UnityEngine.Debug.Log("No Confusions!");
            return;
        }

        foreach (var item in confusions)
        {
            UnityEngine.Debug.Log(
                $"{item.Actual} -> {item.Predicted} : {item.Count}");
        }

        UnityEngine.Debug.Log($"Accuracy: {eval.GetAccuracy(feature):P2}");
    }
}

public class EvaluationResult
{
    public int Total { get; set; }

    public Dictionary<SlotFeature, int> Correct { get; } = new();

    // [실제 정답][예측 결과] = 횟수
    public Dictionary<SlotFeature, Dictionary<string, Dictionary<string, int>>>
        ConfusionMatrices
    { get; } = new();

    public void Add(
        SlotFeature feature,
        string actual,
        string predicted)
    {
        Total++;

        if (actual == predicted)
        {
            Correct[feature] = Correct.GetValueOrDefault(feature) + 1;
        }

        if (!ConfusionMatrices.ContainsKey(feature))
            ConfusionMatrices[feature] = new();

        var matrix = ConfusionMatrices[feature];

        if (!matrix.ContainsKey(actual))
            matrix[actual] = new();

        matrix[actual][predicted] =
            matrix[actual].GetValueOrDefault(predicted) + 1;
    }

    public double GetAccuracy(SlotFeature feature)
    {
        return Total == 0 ? 0 : (double)Correct.GetValueOrDefault(feature) / Total;
    }
}