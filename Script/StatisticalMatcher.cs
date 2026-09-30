using System;
using System.Collections.Generic;
using System.Linq;

public class StatisticalMatcher
{
    // =========================================================
    // Components
    // =========================================================

    private readonly List<ActionTrainingSample> samples =
        new List<ActionTrainingSample>();

    private readonly FeatureScorer featureScorer =
        new FeatureScorer();

    private readonly SlotMatcher slotMatcher;

    private readonly MatchDecision decision =
        new MatchDecision();

    // =========================================================
    // Weights
    // =========================================================

    public float IntendWeight { get; set; } = 1.5f;

    public float TargetWeight { get; set; } = 1.5f;

    public float CharacterWeight { get; set; } = 1.0f;

    public float WordWeight { get; set; } = 1.0f;

    public float ContextWeight { get; set; } = 0.25f;

    public float PersonalWeight { get; set; } = 0.15f;

    public float FocusWeight { get; set; } = 0.75f;

    // =========================================================
    // Constructor
    // =========================================================

    public StatisticalMatcher()
    {
        slotMatcher =
            new SlotMatcher(
                featureScorer);

        slotMatcher.CharacterWeight =
            CharacterWeight;

        slotMatcher.WordWeight =
            WordWeight;
    }

    // =========================================================
    // Calibration
    // =========================================================

    public float SlotExplicitThreshold
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
    }

    // =========================================================
    // Build
    // =========================================================

    public void Build(
        ActionDatabase database)
    {
        samples.Clear();

        foreach (
            ActionDefinition definition
            in database.actions)
        {
            ActionTrainingSample sample =
                new ActionTrainingSample();

            sample.ActionId =
                definition.action_id;

            sample.Intend =
                string.IsNullOrWhiteSpace(
                    definition.intend)
                    ? null
                    : definition.intend;

            sample.Target =
                string.IsNullOrWhiteSpace(
                    definition.target)
                    ? null
                    : definition.target;

            sample.DisplayText =
                definition.display_text;

            foreach (
                string text
                in definition.positive)
            {
                AddFeatures(
                    sample.Positive,
                    text.Normalize(),
                    1f);
            }

            foreach (
                string text
                in definition.negative)
            {
                AddFeatures(
                    sample.Negative,
                    text.Normalize(),
                    1f);
            }

            NormalizeDictionary(
                sample.Positive);

            NormalizeDictionary(
                sample.Negative);

            samples.Add(sample);
        }

        featureScorer.Build(
            samples);
    }

    // =========================================================
    // Match
    // =========================================================

    public List<MatchResult> Match(
        string currentInput,
        string previousActionId = null,
        string focusTarget = null,
        int topK = 5)
    {
        if (string.IsNullOrWhiteSpace(
                currentInput))
        {
            return new List<MatchResult>();
        }

        string current =
            currentInput.Normalize();

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
        // Action candidates
        // -----------------------------------------

        List<MatchResult> results =
            new List<MatchResult>();

        foreach (
            ActionTrainingSample sample
            in samples)
        {
            float intendScore =
                GetSlotScore(
                    intent,
                    sample.NormalizedIntend);

            float targetScore =
                GetSlotScore(
                    target,
                    sample.NormalizedTarget);

            // -----------------------------------------
            // Explicit slot conflict
            // -----------------------------------------

            if (intent.Explicit &&
                !string.Equals(
                    sample.NormalizedIntend,
                    intent.BestLabel,
                    StringComparison.OrdinalIgnoreCase))
            {
                intendScore *= 0.10f;
            }

            if (target.Explicit &&
                !string.Equals(
                    sample.NormalizedTarget,
                    target.BestLabel,
                    StringComparison.OrdinalIgnoreCase))
            {
                targetScore *= 0.10f;
            }

            // -----------------------------------------
            // Features
            // -----------------------------------------

            float characterScore =
                featureScorer.ScoreCharacter(
                    sample,
                    current);

            float wordScore =
                featureScorer.ScoreWord(
                    sample,
                    current);

            // -----------------------------------------
            // Context
            // -----------------------------------------

            float contextScore = 0f;

            // -----------------------------------------
            // Personal
            // -----------------------------------------

            float personalScore =
                ScorePersonal(
                    sample.ActionId,
                    current);

            // -----------------------------------------
            // Focus
            // -----------------------------------------

            float focusScore = 0f;

            bool focusUsed = false;

            if (!target.Explicit &&
                !string.IsNullOrWhiteSpace(
                    focusTarget) &&
                string.Equals(
                    sample.NormalizedTarget,
                    focusTarget,
                    StringComparison.OrdinalIgnoreCase))
            {
                focusScore = 1f;
                focusUsed = true;
            }

            // -----------------------------------------
            // Final
            // -----------------------------------------

            float finalScore =
                intendScore *
                IntendWeight

                +

                targetScore *
                TargetWeight

                +

                characterScore *
                CharacterWeight

                +

                wordScore *
                WordWeight

                +

                contextScore *
                ContextWeight

                +

                personalScore *
                PersonalWeight

                +

                focusScore *
                FocusWeight;

            results.Add(
                new MatchResult(
                    sample.ActionId,

                    sample.Intend,
                    sample.Target,

                    sample.DisplayText,

                    intendScore,
                    targetScore,

                    characterScore,
                    wordScore,

                    contextScore,
                    personalScore,
                    focusScore,

                    finalScore,

                    intent.Explicit,
                    target.Explicit,

                    focusUsed));
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
        List<MatchResult> results)
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

        if (result.Scores.TryGetValue(
                label,
                out float score))
        {
            return score;
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

        float max =
            dictionary.Values.Max();

        if (max <= 0f)
            return;

        List<string> keys =
            dictionary.Keys.ToList();

        foreach (string key in keys)
        {
            dictionary[key] /=
                max;
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
}