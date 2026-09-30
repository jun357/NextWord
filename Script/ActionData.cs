using System;
using System.Collections.Generic;

[Serializable]
public class ActionDefinition
{
    public string action_id;

    public string intend;
    public string target;

    public string display_text;

    public List<string> positive =
        new List<string>();

    public List<string> negative =
        new List<string>();
}

[Serializable]
public class ActionDatabase
{
    public List<ActionDefinition> actions = new List<ActionDefinition>();
}
