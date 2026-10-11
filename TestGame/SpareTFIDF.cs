using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NaturalLanguageActionClassifier
{
    [Serializable]
    public sealed class SparseVector
    {
        public int[] Indices;
        public double[] Values;
    }

    [Serializable]
    public class TextVectorizer
    {
        [Newtonsoft.Json.JsonProperty] private Dictionary<string, int> vocabulary = new();

        [Newtonsoft.Json.JsonProperty] private double[] idf;

        public const int maxFeatures = 30000;

        public const int MinNGram = 2;
        public const int MaxNGram = 4;

        public int VocabularySize => vocabulary.Count;

        public void Fit(List<string> texts)
        {
            vocabulary.Clear();

            var documentFrequency =
                new Dictionary<string, int>();

            foreach (string text in texts)
            {
                var grams = ExtractNGrams(text).Distinct();

                foreach (string gram in grams)
                {
                    documentFrequency.TryGetValue(
                        gram, out int count);

                    documentFrequency[gram] = count + 1;
                }
            }

            var selected = documentFrequency
                .OrderByDescending(x => x.Value)
                .ThenBy(x => x.Key, StringComparer.Ordinal)
                .Take(maxFeatures)
                .ToList();

            for (int i = 0; i < selected.Count; i++)
                vocabulary[selected[i].Key] = i;

            idf = new double[vocabulary.Count];

            for (int i = 0; i < selected.Count; i++)
            {
                int df = selected[i].Value;

                idf[i] =
                    Math.Log(
                        (texts.Count + 1.0) / (df + 1.0)
                    ) + 1.0;
            }
        }

        public SparseVector TransformSparse(string text)
        {
            var counts = new Dictionary<int, int>();

            var grams = ExtractNGrams(text);

            foreach (string gram in grams)
            {
                if (!vocabulary.TryGetValue(
                    gram, out int index))
                    continue;

                counts.TryGetValue(index, out int count);
                counts[index] = count + 1;
            }

            var indices = counts.Keys.OrderBy(x => x).ToArray();
            var values = new double[indices.Length];

            double normSquared = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                int index = indices[i];

                double tf = (double)counts[index];

                values[i] = tf * idf[index];
                normSquared += values[i] * values[i];
            }

            double norm = Math.Sqrt(normSquared);

            if (norm > 0)
            {
                for (int i = 0; i < values.Length; i++)
                    values[i] /= norm;
            }

            return new SparseVector
            {
                Indices = indices,
                Values = values
            };
        }

        private List<string> ExtractNGrams(string text)
        {
            text = Normalize(text);

            var result = new List<string>();

            for (int n = MinNGram;
                 n <= MaxNGram && n <= text.Length;
                 n++)
            {
                for (int i = 0; i <= text.Length - n; i++)
                    result.Add(text.Substring(i, n));
            }

            return result;
        }

        private string Normalize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return "";

            text = text.Trim().ToLowerInvariant();

            var sb = new StringBuilder();

            foreach (char c in text)
            {
                if (char.IsLetterOrDigit(c) ||
                    (c >= 0xAC00 && c <= 0xD7A3) ||
                    c == ' ')
                {
                    sb.Append(c);
                }
            }

            return string.Join(
                " ",
                sb.ToString().Split(
                    new[] { ' ' },
                    StringSplitOptions.RemoveEmptyEntries));
        }
    }
}