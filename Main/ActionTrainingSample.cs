using System.Collections.Generic;

public class ActionTrainingSample
{
    public const string NULL_LABEL = "NULL";

    public string ActionId;

    public string Intend;
    public string Target;

    public string DisplayText;

    public Dictionary<string, float> Positive =
        new Dictionary<string, float>();

    public Dictionary<string, float> Negative =
        new Dictionary<string, float>();

    public string NormalizedIntend
    {
        get
        {
            return string.IsNullOrWhiteSpace(Intend)
                ? NULL_LABEL
                : Intend;
        }
    }

    public string NormalizedTarget
    {
        get
        {
            return string.IsNullOrWhiteSpace(Target)
                ? NULL_LABEL
                : Target;
        }
    }
}