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
    Context = 2
}

[Serializable]
public readonly struct StatisticalFeature
{
    private readonly Dictionary<SlotFeature, string> _slotFeature;
    public string SlotFeatures(SlotFeature feature) => _slotFeature.GetValueOrDefault(feature);

    //private readonly string intend;
    public string Intend => _slotFeature.GetValueOrDefault(SlotFeature.Intend) ?? Normalizer.NULL_LABEL;
    //private readonly string target;
    public string Target => _slotFeature.GetValueOrDefault(SlotFeature.Target) ?? Normalizer.NULL_LABEL;
    //private readonly string context;
    public string Context => _slotFeature.GetValueOrDefault(SlotFeature.Context) ?? Normalizer.NULL_LABEL;

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

    public float[] weights = { 1f,1f, 1f,1f, 1f,1f, 1f,1f };

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

    private void BuildSlotFeature(Dictionary<(SlotFeature, string), SlotFeatureSample> slotfeatureCache, (SlotFeature, string) key, IReadOnlyList<string> ngram, IReadOnlyList<string> words)
    {
        if (!slotfeatureCache.TryGetValue(key, out SlotFeatureSample sample_i))
        {
            sample_i = new(key.Item2);
            slotfeatureCache[key] = sample_i;
        }

        sample_i.Populate(ngram, words);
    }

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

    public (string, string[], double) Predict(string input, int top = 5)
    {
        var stopwatch = Stopwatch.StartNew();
        Vector2[] ws = weights.ToVector2Array();
        var result = Match(input, ws[0], ws[1], ws[2], ws[3], top);
        stopwatch.Stop();
        if (result == null || result.Count == 0)
            return (Normalizer.NULL_ID, System.Array.Empty<string>(), stopwatch.Elapsed.TotalMilliseconds);
        return (result[0].ActionId, result.Select(x => x.ToString()).ToArray(), stopwatch.Elapsed.TotalMilliseconds);
    }

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

    public MatchState Classify(
        IReadOnlyList<MatchResult> results)
    {
        return decision.Classify(results);
    }

    public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
    {
        (List<TrainingSample> dataSet, List<TrainingSample> trainSet) = trainSamples.Split(42, 1 - 0.8235294117647059f);
        Build(dataSet, feature_table);
        optimizer.OptimizeWeightsHacker(trainSet, validationSamples);
    }

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

    private float[] Softmax(float[] logits)
    {
        float maxLogit = logits.Max();
        float[] probabilities = new float[logits.Length];
        float sumExp = 0f;

        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = MathF.Exp(logits[i] - maxLogit);
            sumExp += probabilities[i];
        }

        for (int i = 0; i < probabilities.Length; i++)
            probabilities[i] /= sumExp;

        return probabilities;
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

    private float CalculateCrossEntropy(
    float[] probabilities,
    string[] actionIds,
    string targetActionId)
    {
        int targetIndex = Array.IndexOf(actionIds, targetActionId);

        if (targetIndex < 0)
            throw new ArgumentException(
                "Target action is not present in candidates.",
                nameof(targetActionId));

        return -MathF.Log(
            MathF.Max(probabilities[targetIndex], 1e-7f));
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
}