using System.Text.RegularExpressions;

public static class TextNormalizer
{
    public static string Normalize(this string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return "";

        text = text.Trim()
            .ToLowerInvariant();

        // 모든 연속 whitespace를 한 칸으로
        text = Regex.Replace(
            text,
            @"\s+",
            " ");

        // 흔한 반복 입력 정리
        while (text.Contains("ㅋㅋㅋ"))
            text = text.Replace("ㅋㅋㅋ", "ㅋㅋ");

        while (text.Contains("!!!"))
            text = text.Replace("!!!", "!!");

        while (text.Contains("???"))
            text = text.Replace("???", "??");

        return text;
    }
}