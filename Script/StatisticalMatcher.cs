using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

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
    [SerializeField] private UsedModel usedModel;
    [SerializeField] private WeightOptimizer optimizer;

    // =========================================================
    // Components
    // =========================================================

    private readonly List<ActionTrainingSample> samples = new();
    private readonly Dictionary<(SlotFeature, string), int> keyset = new();
    //private readonly Dictionary<(SlotFeature, string), SlotFeatureSample> SlotFeatureSamples = new();
    //private readonly List<SlotFeatureSample> TargetSamples = new();
    //private readonly List<SlotFeatureSample> ContextSamples = new();

    private readonly FeatureScorer featureScorer = new();

    private SlotMatcher slotMatcher;

    private readonly MatchDecision decision = new();

    // =========================================================
    // Weights
    // =========================================================

    public float[] weights = { 1f,1f, 1f,1f, 1f,1f, 1f,1f };

    /*private StatisticalWeight? _weight;
    private StatisticalWeight Weight
    {
        get
        {
            if (_weight == null)
            {
                if (pre_trained == null)
                {
                    _weight = new(1f, 1f, 0, 1f, 1f);
                }
                else
                {
                    _weight = JsonConvert.DeserializeObject<StatisticalWeight>(pre_trained.text);
                }
            }

            return (StatisticalWeight)_weight;
        }
        set
        {
            _weight = value;
        }
    }

    public float IntendWeight => Weight.intendWeight;

    public float TargetWeight => Weight.targetWeight;

    public float CharacterWeight => Weight.characterWeight;

    public float WordWeight => Weight.wordWeight;*/

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
        //SlotFeatureSamples.Clear();

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
            //BuildSlotFeature(SlotFeatureSamples, (SlotFeature.Intend, sample.Feature.Intend), ngram, words);
            //BuildSlotFeature(SlotFeatureSamples, (SlotFeature.Target, sample.Feature.Target), ngram, words);
            //BuildSlotFeature(SlotFeatureSamples, (SlotFeature.Context, sample.Feature.Context), ngram, words);
        }

        samples.Clear();
        //IntendSamples.Clear();
        //TargetSamples.Clear();
        //ContextSamples.Clear();
        foreach (var sample in sampleCache.Values)
        {
            samples.Add(sample);
        }
        /*foreach ((var key, var sample) in slotfeatureCache)
        {
            switch (key.Item1)
            {
                case SlotFeature.Intend:
                    IntendSamples.Add(sample);
                    break;
                case SlotFeature.Target:
                    TargetSamples.Add(sample);
                    break;
                case SlotFeature.Context:
                    ContextSamples.Add(sample);
                    break;
            }
        }*/

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
                //AddFeatures(sample.charFRQ, sample.wordFRQ, text.Normalize());
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
        //List<float> result = new();

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
            //result.Add(finalScore);

            idx++;
        }

        return max_id;
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
            /*var intendKey = (SlotFeature.Intend, sample.Feature.Intend);
            if (!cache.TryGetValue(intendKey, out float intendScore))
            {
                intendScore = Vector2.Dot(GetSlotScore(scores, index, intendKey), v1);
                cache[intendKey] = intendScore;
            }

            var targetKey = (SlotFeature.Target, sample.Feature.Target);
            if (!cache.TryGetValue(targetKey, out float targetScore))
            {
                targetScore = Vector2.Dot(GetSlotScore(scores, index, targetKey), v2);
                cache[targetKey] = targetScore;
            }

            var contextKey = (SlotFeature.Context, sample.Feature.Context);
            if (!cache.TryGetValue(contextKey, out float contextScore))
            {
                contextScore = Vector2.Dot(GetSlotScore(scores, index, contextKey), v3);
                cache[contextKey] = contextScore;
            }*/

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

    // =========================================================
    // Slot score
    // =========================================================

    /*private Vector2 GetSlotScore(Vector2[] scores, Dictionary<(SlotFeature, string), List<int>> index, (SlotFeature, string) key)
    {
        if (!index.TryGetValue(key, out var list) || list.Count == 0)
        {
            return new(0, 0);
        }

        Vector2 sum = new(0, 0);
        foreach(int idx in list)
        {
            sum += scores[idx]; 
        }

        return sum / list.Count;
    }

    private float GetSlotScore(SlotMatcher.SlotResult result, string label)
    {
        if (result?.Scores == null)
        {
            return 0f;
        }

        if (result.Scores.TryGetValue(label, out (float, int) entry))
        {
            return SlotMatcher.ToScore(entry);
        }

        return 0f;
    }*/

    // =========================================================
    // Feature building
    // =========================================================

    /*private void AddFeatures(Dictionary<string, float> charFRQ, Dictionary<string, float> wordFRQ, string text)
    {
        foreach (string gram in FeatureScorer.ExtractCharacterNGrams(text))
        {
            if (!charFRQ.ContainsKey(gram))
                charFRQ[gram] = 0;

            charFRQ[gram] += 1;
        }

        foreach (string word in FeatureScorer.TokenizeWords(text))
        {
            if (!wordFRQ.ContainsKey(word))
                wordFRQ[word] = 0;

            wordFRQ[word] += 1;
        }
    }

    private void NormalizeDictionary(
        Dictionary<string, int> dictionary)
    {
        if (dictionary.Count == 0)
            return;

        int max = dictionary.Values.Max();

        if (max <= 0f)
            return;

        List<string> keys = dictionary.Keys.ToList();

        foreach (string key in keys)
        {
            dictionary[key] /= max;
        }
    }*/

    public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
    {
        Build(trainSamples, feature_table);
        optimizer.OptimizeWeightsPro(validationSamples);
    }

    public void LoadModel(ActionDatabase database)
    {
        Build(database);
    }
}