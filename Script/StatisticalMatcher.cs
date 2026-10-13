using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

[Serializable]
public readonly struct StatisticalFeature
{
    private readonly string intend;
    public string Intend => intend ?? Normalizer.NULL_LABEL;
    private readonly string target;
    public string Target => target ?? Normalizer.NULL_LABEL;
    private readonly string context;
    public string Context => context ?? Normalizer.NULL_LABEL;

    [JsonConstructor]
    public StatisticalFeature(string Intend, string Target, string Context)
    {
        intend = Intend;
        target = Target;
        context = Context;
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

    private readonly FeatureScorer featureScorer = new FeatureScorer();

    private SlotMatcher slotMatcher;

    private readonly MatchDecision decision = new MatchDecision();

    // =========================================================
    // Weights
    // =========================================================

    private StatisticalWeight? _weight;
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

    public float WordWeight => Weight.wordWeight;

    public bool WasPreTrained => pre_trained != null;

    //public float ContextWeight { get; set; } = 0.25f;

    //public float PersonalWeight { get; set; } = 0.15f;

    //public float FocusWeight { get; set; } = 0.75f;

    // =========================================================
    // Constructor
    // =========================================================

    /*public StatisticalMatcher(ActionDatabase database)
    {
        slotMatcher = new (featureScorer);
        slotMatcher.CharacterWeight = CharacterWeight;
        slotMatcher.WordWeight = WordWeight;

        Build(database);
    }*/

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
    // Calibration
    // =========================================================

    /*public float SlotExplicitThreshold
    {
        get => slotMatcher.ExplicitThreshold;
        set => slotMatcher.ExplicitThreshold = value;
    }

    public float SlotExplicitGap
    {
        get => slotMatcher.ExplicitGap;
        set => slotMatcher.ExplicitGap = value;
    }

    public float AmbiguousRelativeGap
    {
        get => decision.AmbiguousRelativeGap;
        set => decision.AmbiguousRelativeGap = value;
    }*/

    // =========================================================
    // Build
    // =========================================================

    void ITrain.Train(ActionDatabase database) => Build(database);

    private void Build(List<TrainingSample> trainSamples, Dictionary<string, StatisticalFeature> feature_table)
    {
        MakeSlotMatcher();

        Dictionary<string, ActionTrainingSample> sampleCache = new();

        foreach (TrainingSample trainSample in trainSamples)
        {
            if (!sampleCache.TryGetValue(trainSample.Action, out ActionTrainingSample sample))
            {
                sample = new(trainSample.Action, feature_table.GetValueOrDefault(trainSample.Action));
                sampleCache[trainSample.Action] = sample;
            }

            AddFeatures(sample.Positive, trainSample.Text.Normalize(), 1f);

            //samples.Add(sample);
        }

        samples.Clear();
        foreach (var sample in sampleCache.Values)
        {
            NormalizeDictionary(sample.Positive);
            samples.Add(sample);
        }

        featureScorer.Build(samples);
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

            foreach (
                string text
                in definition.positive)
            {
                AddFeatures(
                    sample.Positive,
                    text.Normalize(),
                    1f);
            }

            /*foreach (
                string text
                in definition.negative)
            {
                AddFeatures(
                    sample.Negative,
                    text.Normalize(),
                    1f);
            }*/

            NormalizeDictionary(
                sample.Positive);

            //NormalizeDictionary(sample.Negative);

            samples.Add(sample);
        }

        featureScorer.Build(
            samples);
    }

    public (string, string[], double) Predict(string input, int top = 5)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = Match(input, Weight, top);
        stopwatch.Stop();
        return (result[0].ActionId, result.Select(x => x.ToString()).ToArray(), stopwatch.Elapsed.TotalMilliseconds);
    }

    // =========================================================
    // Match
    // =========================================================

    public IReadOnlyList<MatchResult> Match(
        string currentInput,
        StatisticalWeight weight,
        int topK = 5)
    {
        if (string.IsNullOrWhiteSpace(currentInput))
        {
            return Array.Empty<MatchResult>();
        }

        string current = currentInput.Normalize();

        // -----------------------------------------
        // INTENT
        // -----------------------------------------

        SlotMatcher.SlotResult intent =
            slotMatcher.MatchIntent(
                samples,
                current);

        // -----------------------------------------
        // TARGET
        // -----------------------------------------

        SlotMatcher.SlotResult target =
            slotMatcher.MatchTarget(
                samples,
                current);

        // -----------------------------------------
        // TARGET
        // -----------------------------------------

        SlotMatcher.SlotResult context =
            slotMatcher.MatchContext(
                samples,
                current);

        // -----------------------------------------
        // Action candidates
        // -----------------------------------------

        List<MatchResult> results = new();

        foreach (ActionTrainingSample sample in samples)
        {
            float intendScore = GetSlotScore(intent, sample.Feature.Intend);
            float targetScore = GetSlotScore(target, sample.Feature.Target);
            float contextScore = GetSlotScore(context, sample.Feature.Context);

            float characterScore =
                featureScorer.ScoreCharacter(
                    sample,
                    current);

            float wordScore =
                featureScorer.ScoreWord(
                    sample,
                    current);

            float finalScore =
            intendScore * weight.intendWeight

            + targetScore * weight.targetWeight

            + contextScore * weight.contextWeight

            + characterScore * weight.characterWeight

            + wordScore * weight.wordWeight
            ;

            results.Add(
                new MatchResult(
                    sample.ActionId,

                    sample.Feature,

                    intendScore,
                    targetScore,
                    contextScore,

                    characterScore,
                    wordScore,

                    finalScore,

                    intent.Explicit,
                    target.Explicit
                    ));
        }

        return results
            .OrderByDescending(
             x => x.FinalScore)
            .Take(topK)
            .ToList();
    }

    // =========================================================
    // Classify
    // =========================================================

    public MatchState Classify(
        IReadOnlyList<MatchResult> results)
    {
        return decision.Classify(
            results);
    }

    // =========================================================
    // Slot score
    // =========================================================

    private float GetSlotScore(
        SlotMatcher.SlotResult result,
        string label)
    {
        if (result == null ||
            result.Scores == null)
        {
            return 0f;
        }

        if (result.Scores.TryGetValue(label, out (float, int) entry))
        {
            return SlotMatcher.ToScore(entry);
        }

        return 0f;
    }

    // =========================================================
    // Feature building
    // =========================================================

    private void AddFeatures(
        Dictionary<string, float> target,
        string text,
        float weight)
    {
        foreach (
            string gram
            in featureScorer.ExtractCharacterNGrams(
                text))
        {
            if (!target.ContainsKey(gram))
                target[gram] = 0f;

            target[gram] += weight;
        }

        foreach (
            string word
            in featureScorer.TokenizeWords(
                text))
        {
            if (!target.ContainsKey(word))
                target[word] = 0f;

            target[word] += weight;
        }
    }

    private void NormalizeDictionary(
        Dictionary<string, float> dictionary)
    {
        if (dictionary.Count == 0)
            return;

        float max = dictionary.Values.Max();

        if (max <= 0f)
            return;

        List<string> keys = dictionary.Keys.ToList();

        foreach (string key in keys)
        {
            dictionary[key] /= max;
        }
    }

    // =========================================================
    // Personal
    // =========================================================

    private readonly Dictionary<
        string,
        Dictionary<string, float>>
        personalWeights =
        new Dictionary<
            string,
            Dictionary<string, float>>();

    private float ScorePersonal(
        string actionId,
        string input)
    {
        if (!personalWeights.TryGetValue(
                actionId,
                out Dictionary<string, float> map))
        {
            return 0f;
        }

        float score = 0f;

        foreach (
            string gram
            in featureScorer.ExtractCharacterNGrams(
                input))
        {
            if (map.TryGetValue(
                    gram,
                    out float value))
            {
                score += value;
            }
        }

        return score;
    }

    public void AddPositiveFeedback(
        string actionId,
        string input)
    {
        if (string.IsNullOrWhiteSpace(
                actionId))
        {
            return;
        }

        if (!personalWeights.ContainsKey(
                actionId))
        {
            personalWeights[actionId] =
                new Dictionary<string, float>();
        }

        Dictionary<string, float> map =
            personalWeights[actionId];

        foreach (
            string gram
            in featureScorer.ExtractCharacterNGrams(
                input.Normalize()))
        {
            if (!map.ContainsKey(gram))
                map[gram] = 0f;

            map[gram] += 1f;
        }
    }

    public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
    {
        Build(trainSamples, feature_table);
        optimizer.OptimizeWeights(validationSamples);
    }

    public void LoadModel(ActionDatabase database)
    {
        Build(database);
    }
}