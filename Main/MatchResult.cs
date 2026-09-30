public class MatchResult
{
    public string ActionId;

    public string Intend;
    public string Target;

    public string DisplayText;

    public float IntendScore;
    public float TargetScore;

    public float CharacterScore;
    public float WordScore;

    public float ContextScore;
    public float PersonalScore;
    public float FocusScore;

    public float FinalScore;

    public bool IntendExplicit;
    public bool TargetExplicit;

    public bool FocusUsed;

    public MatchResult(
        string actionId,
        string intend,
        string target,
        string displayText,

        float intendScore,
        float targetScore,

        float characterScore,
        float wordScore,

        float contextScore,
        float personalScore,
        float focusScore,

        float finalScore,

        bool intendExplicit,
        bool targetExplicit,

        bool focusUsed)
    {
        ActionId = actionId;

        Intend = intend;
        Target = target;

        DisplayText = displayText;

        IntendScore = intendScore;
        TargetScore = targetScore;

        CharacterScore = characterScore;
        WordScore = wordScore;

        ContextScore = contextScore;
        PersonalScore = personalScore;
        FocusScore = focusScore;

        FinalScore = finalScore;

        IntendExplicit = intendExplicit;
        TargetExplicit = targetExplicit;

        FocusUsed = focusUsed;
    }

    public override string ToString()
    {
        return
            $"ACTION={(string.IsNullOrWhiteSpace(ActionId) ? "NULL_ACTION" : ActionId)} / " +
            $"INTENT={Intend ?? "NULL"}({IntendScore:F3}) / " +
            $"TARGET={Target ?? "NULL"}({TargetScore:F3}) / " +
            $"Character={CharacterScore:F3} / " +
            $"Word={WordScore:F3} / " +
            $"Context={ContextScore:F3} / " +
            $"Personal={PersonalScore:F3} / " +
            $"Focus={FocusScore:F3} / " +
            $"Final={FinalScore:F3} / " +
            $"Explicit=I:{IntendExplicit},T:{TargetExplicit} / " +
            $"FocusUsed={FocusUsed} / " +
            $"Display={DisplayText}";
    }
}