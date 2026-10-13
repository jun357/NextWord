using System;
using System.Collections.Generic;

[Serializable]
public class ActionDefinition
{
    public string action_id;

    public string intend = Normalizer.NULL_LABEL;
    public string target = Normalizer.NULL_LABEL;
    public string context = Normalizer.NULL_LABEL;

    public List<string> positive = new();
}

[Serializable]
public class ActionDatabase
{
    public List<ActionDefinition> actions = new();
}