using TMPro;
using UnityEngine;

public class PlayGame : MonoBehaviour
{
    private EscapeRoom escapeRoom = new();
    //[SerializeField] private PlayerAgent playerAgent;
    [SerializeField] private TMP_InputField output;
    [SerializeField] private InputConsumer inputConsumer;

    private void Awake()
    {
        inputConsumer.OnPredict += Respond;
    }

    private void Start()
    {
        output.text = escapeRoom.OnGameStart;
    }

    private void Respond(EscapeAction action)
    {
        output.text = escapeRoom.Execute(action);
    }
}