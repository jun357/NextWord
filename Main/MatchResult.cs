public class MatchResult
{
    public string ActionId;
    public StatisticalFeature Feature;
    public float IntendScore;
    public float TargetScore;
    public float ContextScore;
    public float ActionScore;
    public float FinalScore;

    public MatchResult(
        string actionId,
        StatisticalFeature feature,
        float intendScore,
        float targetScore,
        float contextScore,
        float actionScore,
        float finalScore
    )
    {
        ActionId = actionId;
        Feature = feature;
        IntendScore = intendScore;
        TargetScore = targetScore;
        ActionScore = actionScore;
        ContextScore = contextScore;
        FinalScore = finalScore;
    }

    public override string ToString()
    {
        return
            $"ACTION={ActionId}({ActionScore:F3}) / " +
            $"INTENT={Feature.Intend ?? "NULL"}({IntendScore:F3}) / " +
            $"TARGET={Feature.Target ?? "NULL"}({TargetScore:F3}) / " +
            $"CONTEXT={Feature.Context ?? "NULL"}({ContextScore:F3}) / " +
            $"Final={FinalScore:F3} / ";
    }
}