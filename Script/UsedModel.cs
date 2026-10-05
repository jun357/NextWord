using UnityEngine;

[CreateAssetMenu(fileName = "UsedModel", menuName = "Scriptable Objects/UsedModel")]
public class UsedModel : ScriptableObject
{
    public ITrain Model { get; set; }
}
