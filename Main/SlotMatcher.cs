using System;
using System.Collections.Generic;
using System.Linq;


public class SlotMatcher
{
    public const string NULL_LABEL = "NULL";

    public class SlotResult
    {
        public string BestLabel;

        public float BestScore;

        public float SecondScore;

        public float RelativeGap;

        public bool Explicit;

        public Dictionary<string, (float, int)> Scores = new();

        public SlotResult()
        {
        }
    }

    public float CharacterWeight { get; set; } = 1f;
    public float WordWeight { get; set; } = 1f;

    // 실제 evidence 기준
    public float ExplicitThreshold { get; set; } = 0.20f;

    // 1등이 2등보다 얼마나 우세해야 하는가
    public float ExplicitGap { get; set; } = 0.10f;

    private readonly FeatureScorer scorer;

    public SlotMatcher(
        FeatureScorer scorer)
    {
        this.scorer = scorer;
    }

    // =========================================================
    // INTENT
    // =========================================================

    public SlotResult MatchIntent(
        IList<ActionTrainingSample> samples,
        string input)
    {
        return Match(samples, input, x => x.Feature.Intend);
    }

    // =========================================================
    // TARGET
    // =========================================================

    public SlotResult MatchTarget(
        IList<ActionTrainingSample> samples,
        string input)
    {
        return Match(samples, input, x => x.Feature.Target);
    }

    // =========================================================
    // CONTEXT
    // =========================================================

    public SlotResult MatchContext(
        IList<ActionTrainingSample> samples,
        string input)
    {
        return Match(samples, input, x => x.Feature.Context);
    }

    // =========================================================
    // Generic slot matching
    // =========================================================

    private SlotResult Match(
        IList<ActionTrainingSample> samples,
        string input,
        Func<ActionTrainingSample, string> labelSelector)
    {
        Dictionary<string, (float, int)> labelScores =
            new Dictionary<string, (float, int)>(
                StringComparer.OrdinalIgnoreCase);

        foreach (ActionTrainingSample sample in samples)
        {
            string label = NormalizeLabel(labelSelector(sample));

            float score =
                scorer.Score(
                    sample,
                    input,
                    CharacterWeight,
                    WordWeight);

            if (!labelScores.ContainsKey(label))
                labelScores[label] = (0f, 0);

            // label을 대표하는 최고 evidence
            //if (score > labelScores[label])
            labelScores[label] = (labelScores[label].Item1 + score, labelScores[label].Item2 + 1)
            // samples.Count
            ;
        }

        return BuildResult(labelScores);
    }

    // =========================================================
    // Result
    // =========================================================

    public static float ToScore((float, int) entry)
    {
        return entry.Item2 == 0 ? 0 : entry.Item1 / entry.Item2;
    }

    private SlotResult BuildResult(
        Dictionary<string, (float, int)> scores)
    {
        SlotResult result = new();

        result.Scores = scores;

        if (scores.Count == 0)
        {
            result.BestLabel = NULL_LABEL;

            return result;
        }

        List<KeyValuePair<string, (float, int)>> ordered = scores.OrderByDescending(x => ToScore(x.Value)).ToList();

        result.BestLabel = ordered[0].Key;

        result.BestScore = ToScore(ordered[0].Value);

        result.SecondScore = ordered.Count >= 2 ? ToScore(ordered[1].Value) : 0f;

        if (result.BestScore > 0f)
        {
            result.RelativeGap =
                (result.BestScore -
                 result.SecondScore)
                /
                result.BestScore;
        }

        bool notNull =
            result.BestLabel != NULL_LABEL;

        bool enoughEvidence =
            result.BestScore >=
            ExplicitThreshold;

        bool enoughSeparation =
            result.RelativeGap >=
            ExplicitGap;

        result.Explicit =
            notNull &&
            enoughEvidence &&
            enoughSeparation;

        return result;
    }

    private string NormalizeLabel(
        string label)
    {
        return string.IsNullOrWhiteSpace(label)
            ? NULL_LABEL
            : label;
    }
}