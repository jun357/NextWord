using System;
using UnityEngine;

[CreateAssetMenu(fileName = "UsedModel", menuName = "Scriptable Objects/UsedModel")]
public class UsedModel : ScriptableObject
{
    private ITrain _model;
    public ITrain Model 
    { 
        get => _model;
        set 
        {
            _model = value;
            OnModelLoad?.Invoke();
        }
    }
    public event Action OnModelLoad;
}
