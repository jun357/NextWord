using System;
using System.Collections.Generic;
using System.Linq;

public class FeatureScorer
{
    public int CharacterNGram { get; set; } = 2;

    private readonly Dictionary<string, int>
        characterDocumentFrequency =
        new Dictionary<string, int>();

    private readonly Dictionary<string, int>
        wordDocumentFrequency =
        new Dictionary<string, int>();

    private int documentCount;

    // =========================================================
    // Build
    // =========================================================

    public void Build(
        IList<ActionTrainingSample> samples)
    {
        characterDocumentFrequency.Clear();
        wordDocumentFrequency.Clear();

        documentCount = samples.Count;

        foreach (ActionTrainingSample sample in samples)
        {
            HashSet<string> characterUnique =
                new HashSet<string>(
                    sample.Positive.Keys
                        .Where(x => x.Length == CharacterNGram));

            foreach (string gram in characterUnique)
            {
                if (!characterDocumentFrequency.ContainsKey(gram))
                    characterDocumentFrequency[gram] = 0;

                characterDocumentFrequency[gram]++;
            }

            HashSet<string> wordUnique =
                new HashSet<string>(
                    sample.Positive.Keys
                        .Where(x => x.Length != CharacterNGram));

            foreach (string word in wordUnique)
            {
                if (!wordDocumentFrequency.ContainsKey(word))
                    wordDocumentFrequency[word] = 0;

                wordDocumentFrequency[word]++;
            }
        }
    }

    // =========================================================
    // Character
    // =========================================================

    public float ScoreCharacter(
        ActionTrainingSample sample,
        string input)
    {
        List<string> grams =
            ExtractCharacterNGrams(input);

        if (grams.Count == 0)
            return 0f;

        Dictionary<string, int> tf =
            BuildTermFrequency(grams);

        float positive = 0f;
        float negative = 0f;

        foreach (KeyValuePair<string, int> pair in tf)
        {
            string gram = pair.Key;

            float tfValue = pair.Value;

            float idf =
                CharacterIDF(gram);

            if (sample.Positive.TryGetValue(
                    gram,
                    out float p))
            {
                positive +=
                    tfValue *
                    p *
                    idf;
            }

            /*if (sample.Negative.TryGetValue(
                    gram,
                    out float n))
            {
                negative +=
                    tfValue *
                    n *
                    idf;
            }*/
        }

        return Math.Max(
            0f,
            positive - negative);
    }

    // =========================================================
    // Word
    // =========================================================

    public float ScoreWord(
        ActionTrainingSample sample,
        string input)
    {
        string[] words =
            TokenizeWords(input);

        if (words.Length == 0)
            return 0f;

        Dictionary<string, int> tf =
            BuildTermFrequency(words);

        float positive = 0f;
        float negative = 0f;

        foreach (KeyValuePair<string, int> pair in tf)
        {
            string word = pair.Key;

            float tfValue = pair.Value;

            float idf =
                WordIDF(word);

            if (sample.Positive.TryGetValue(
                    word,
                    out float p))
            {
                positive +=
                    tfValue *
                    p *
                    idf;
            }

            /*if (sample.Negative.TryGetValue(
                    word,
                    out float n))
            {
                negative +=
                    tfValue *
                    n *
                    idf;
            }*/
        }

        return Math.Max(
            0f,
            positive - negative);
    }

    // =========================================================
    // Combined
    // =========================================================

    public float Score(
        ActionTrainingSample sample,
        string input,
        float characterWeight,
        float wordWeight)
    {
        return
            ScoreCharacter(sample, input)
                * characterWeight
            +
            ScoreWord(sample, input)
                * wordWeight;
    }

    // =========================================================
    // Feature extraction
    // =========================================================

    public List<string> ExtractCharacterNGrams(
        string text)
    {
        List<string> result =
            new List<string>();

        if (string.IsNullOrEmpty(text))
            return result;

        if (text.Length < CharacterNGram)
            return result;

        for (
            int i = 0;
            i <= text.Length - CharacterNGram;
            i++)
        {
            result.Add(
                text.Substring(
                    i,
                    CharacterNGram));
        }

        return result;
    }

    public string[] TokenizeWords(
        string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<string>();

        return text
            .Split(
                new char[]
                {
                    ' ',
                    '\t',
                    '\r',
                    '\n',
                    ',',
                    '.',
                    '!',
                    '?',
                    '。',
                    '！',
                    '？'
                },
                StringSplitOptions.RemoveEmptyEntries);
    }

    private Dictionary<string, int>
        BuildTermFrequency(
            IEnumerable<string> terms)
    {
        Dictionary<string, int> tf =
            new Dictionary<string, int>();

        foreach (string term in terms)
        {
            if (!tf.ContainsKey(term))
                tf[term] = 0;

            tf[term]++;
        }

        return tf;
    }

    // =========================================================
    // IDF
    // =========================================================

    private float CharacterIDF(
        string gram)
    {
        if (!characterDocumentFrequency.TryGetValue(
                gram,
                out int df))
        {
            return 1f;
        }

        return
            (float)Math.Log(
                (documentCount + 1f) /
                (df + 1f))
            + 1f;
    }

    private float WordIDF(
        string word)
    {
        if (!wordDocumentFrequency.TryGetValue(
                word,
                out int df))
        {
            return 1f;
        }

        return
            (float)Math.Log(
                (documentCount + 1f) /
                (df + 1f))
            + 1f;
    }
}