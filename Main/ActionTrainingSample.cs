using System.Collections.Generic;

public static class Normalizer
{
    public const string NULL_ID = "NoAction";
    public const string NULL_LABEL = "NULL";

    public static string Normalize(this string text) => string.IsNullOrWhiteSpace(text) ? NULL_LABEL : text;
}

public sealed class ActionTrainingSample
{
    public string ActionId { get; }
    public StatisticalFeature Feature { get; }

    public int sampleCount;

    public readonly Dictionary<string, float> charFRQ = new();
    public readonly Dictionary<string, float> wordFRQ = new();

    public ActionTrainingSample(string action, StatisticalFeature feature)
    {
        ActionId = action;
        Feature = feature;
    }

    public void Populate(IReadOnlyList<string> ngram, IReadOnlyList<string> words)
    {
        sampleCount++;
        FeatureScorer.PopulateTF(charFRQ, ngram);
        FeatureScorer.PopulateTF(wordFRQ, words);
    }
}

public sealed class SlotFeatureSample
{
    public string Lable { get; }

    //public int sampleCount;

    public readonly Dictionary<string, float> charFRQ = new();
    public readonly Dictionary<string, float> wordFRQ = new();

    public SlotFeatureSample(string label)
    {
        Lable = label;
    }

    public void Populate(IReadOnlyList<string> ngram, IReadOnlyList<string> words)
    {
        //sampleCount++;
        FeatureScorer.PopulateTF(charFRQ, ngram);
        FeatureScorer.PopulateTF(wordFRQ, words);
    }
}