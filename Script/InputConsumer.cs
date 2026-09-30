using TMPro;
using UnityEngine;

public class InputConsumer : MonoBehaviour
{
    [SerializeField]
    private TMP_InputField input;
    [SerializeField]
    private ActionMatcherTester tester;

    public void OnEndEdit()
    {
        tester.Test(input.text);
    }
}
