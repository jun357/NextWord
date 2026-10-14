using System;
using System.Collections.Generic;
using UnityEngine;

public class FeatureScorer
{
    public const int CharacterNGram = 2;

    private readonly Dictionary<string, float> charDF = new();

    private readonly Dictionary<string, float> wordDF = new();

    //private int documentCount;

    // =========================================================
    // Build
    // =========================================================

    public void Build(IList<ActionTrainingSample> samples)
    {
        //documentCount = 0;
        charDF.Clear();
        wordDF.Clear();

        foreach (ActionTrainingSample sample in samples)
        {
            //documentCount += sample.sampleCount;

            foreach ((string gram, float count) in sample.charFRQ)
            {
                if (!charDF.ContainsKey(gram))
                    charDF[gram] = 0;

                charDF[gram] += count;
            }

            foreach ((string word, float count) in sample.wordFRQ)
            {
                if (!wordDF.ContainsKey(word))
                    wordDF[word] = 0;

                wordDF[word] += count;
            }
        }
    }

    // =========================================================
    // Character
    // =========================================================

    /*public float ScoreCharacter(ActionTrainingSample sample, string input)
    {
        List<string> grams = ExtractCharacterNGrams(input);

        if (grams.Count == 0)
            return 0f;

        Dictionary<string, int> tf = BuildTermFrequency(grams);

        float positive = 0f;
        float negative = 0f;

        foreach (KeyValuePair<string, int> pair in tf)
        {
            string gram = pair.Key;

            float tfValue = pair.Value;

            float idf = CharacterIDF(gram);

            if (sample.Positive.TryGetValue(gram, out float p))
            {
                positive += tfValue * p * idf;
            }
        }

        return Math.Max(0f, positive - negative);
    }*/

    public float ScoreCharacter(IReadOnlyDictionary<string, int> tf, IReadOnlyDictionary<string, float> positive)
    {
        float score = 0f;

        foreach ((string gram, int FRQ) in tf)
        {
            if (positive.TryGetValue(gram, out float p))
            {
                score += FRQ * p * CharacterIDF(gram);
            }
        }

        return Math.Max(0f, score);
    }

    // =========================================================
    // Word
    // =========================================================

    /*public float ScoreWord(
        ActionTrainingSample sample,
        string input)
    {
        string[] words = TokenizeWords(input);

        if (words.Length == 0)
            return 0f;

        Dictionary<string, int> tf = BuildTermFrequency(words);

        float positive = 0f;
        float negative = 0f;

        foreach (KeyValuePair<string, int> pair in tf)
        {
            string word = pair.Key;

            float tfValue = pair.Value;

            float idf = WordIDF(word);

            if (sample.Positive.TryGetValue(
                    word,
                    out float p))
            {
                positive +=
                    tfValue *
                    p *
                    idf;
            }

            if (sample.Negative.TryGetValue(
                    word,
                    out float n))
            {
                negative +=
                    tfValue *
                    n *
                    idf;
            }
        }

        return Math.Max(
            0f,
            positive - negative);
    }*/

    public float ScoreWord(IReadOnlyDictionary<string, int> tf, IReadOnlyDictionary<string, float> positive)
    {
        float score = 0f;

        foreach ((string word, float FRQ) in tf)
        {
            if (positive.TryGetValue(word, out float p))
            {
                score += FRQ * p * WordIDF(word);
            }
        }

        return Math.Max(0f, score);
    }

    // =========================================================
    // Combined
    // =========================================================

    /*public float Score(
        ActionTrainingSample sample,
        string input,
        float characterWeight,
        float wordWeight)
    {
        return ScoreCharacter(sample, input) * characterWeight
             + ScoreWord(sample, input) * wordWeight;
    }*/

    /*public float Score(
        IReadOnlyDictionary<string, int> charTf,
        IReadOnlyDictionary<string, int> wordTf, 
        IReadOnlyDictionary<string, float> positive,
        float characterWeight,
        float wordWeight)
    {
        return ScoreCharacter(charTf, positive) * characterWeight
             + ScoreWord(wordTf, positive) * wordWeight;
    }*/

    public Vector2 ExtractScore(
        IReadOnlyDictionary<string, int> charTf,
        IReadOnlyDictionary<string, int> wordTf,
        IReadOnlyDictionary<string, float> charFRQ,
        IReadOnlyDictionary<string, float> wordFRQ)
    {
        return new Vector2(ScoreCharacter(charTf, charFRQ), ScoreWord(wordTf, wordFRQ));
    }

    // =========================================================
    // Feature extraction
    // =========================================================

    public static List<string> ExtractCharacterNGrams(string text)
    {
        List<string> result = new();

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

    public static string[] TokenizeWords(
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

    public static void PopulateTF(Dictionary<string, int> tf, IEnumerable<string> terms)
    {
        foreach (string term in terms)
        {
            if (!tf.ContainsKey(term))
                tf[term] = 0;

            tf[term]++;
        }
    }

    public static void PopulateTF(Dictionary<string, float> tf, IEnumerable<string> terms)
    {
        foreach (string term in terms)
        {
            if (!tf.ContainsKey(term))
                tf[term] = 0;

            tf[term]++;
        }
    }

    // =========================================================
    // IDF
    // =========================================================

    private float CharacterIDF(string gram)
    {
        if (!charDF.TryGetValue(gram, out float df))
        {
            return 1f;
        }

        return 77 / df;
        //return (float)Math.Log((documentCount + 1f) / (df + 1f)) + 1f;
    }

    private float WordIDF(
        string word)
    {
        if (!wordDF.TryGetValue(word, out float df))
        {
            return 1f;
        }

        return 77 / df;
        //return (float)Math.Log((documentCount + 1f) / (df + 1f))+ 1f;
    }
}