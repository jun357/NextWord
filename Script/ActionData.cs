using System;
using System.Collections.Generic;

[Serializable]
public class ActionDefinition
{
    public EscapeAction action_id;

    public string intend;
    public string target;

    public string display_text;

    public List<string> positive = new();

    //public List<string> negative = new();
}

[Serializable]
public class ActionDatabase
{
    public List<ActionDefinition> actions = new();
}