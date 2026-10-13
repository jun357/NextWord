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

    public Dictionary<string, float> Positive = new();

    public ActionTrainingSample(string action, StatisticalFeature feature)
    {
        ActionId = action;
        Feature = feature;
    }
}