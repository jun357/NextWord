public class MatchResult
{
    public string ActionId;

    public StatisticalFeature Feature;

    public string DisplayText;
    
    public float IntendScore;
    public float TargetScore;
    public float ContextScore;

    public float CharacterScore;
    public float WordScore;

    //public float ContextScore;
    //public float PersonalScore;
    //public float FocusScore;

    public float FinalScore;

    public bool IntendExplicit;
    public bool TargetExplicit;

    //public bool FocusUsed;

    public MatchResult(
        string actionId,
        StatisticalFeature feature,

        float intendScore,
        float targetScore,
        float contextScore,

        float characterScore,
        float wordScore,

        //float contextScore,
        //float personalScore,
        //float focusScore,

        float finalScore,

        bool intendExplicit,
        bool targetExplicit

        //bool focusUsed
    )
    {
        ActionId = actionId;

        Feature = feature;

        IntendScore = intendScore;
        TargetScore = targetScore;

        CharacterScore = characterScore;
        WordScore = wordScore;
        ContextScore = contextScore;

        //ContextScore = contextScore;
        //PersonalScore = personalScore;
        //FocusScore = focusScore;

        FinalScore = finalScore;

        IntendExplicit = intendExplicit;
        TargetExplicit = targetExplicit;

        //FocusUsed = focusUsed;
    }

    public override string ToString()
    {
        return
            $"ACTION={ActionId} / " +
            $"INTENT={Feature.Intend ?? "NULL"}({IntendScore:F3}) / " +
            $"TARGET={Feature.Target ?? "NULL"}({TargetScore:F3}) / " +
            $"CONTEXT={Feature.Context ?? "NULL"}({ContextScore:F3}) / " +
            $"Character={CharacterScore:F3} / " +
            $"Word={WordScore:F3} / " +
            //$"Context={ContextScore:F3} / " +
            //$"Personal={PersonalScore:F3} / " +
            //$"Focus={FocusScore:F3} / " +
            $"Final={FinalScore:F3} / " +
            $"Explicit=I:{IntendExplicit},T:{TargetExplicit} / " +
            //$"FocusUsed={FocusUsed} / " +
            $"Display={DisplayText}";
    }
}