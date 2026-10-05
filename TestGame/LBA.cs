using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using UnityEngine;

namespace NaturalLanguageActionClassifier
{
    // ============================================================
    // 1. 학습 데이터
    // ============================================================

    public class TrainingSample
    {
        public string Text { get; }
        public EscapeAction Action { get; }

        public TrainingSample(string text, EscapeAction action)
        {
            Text = text;
            Action = action;
        }
    }

    // ============================================================
    // 2. 텍스트 벡터화
    //
    // 한국어 형태소 분석기 없이 문자 n-gram 사용
    //
    // "불 켜줘"
    // -> "불 "
    // -> " 켜"
    // -> "켜줘"
    // -> ...
    // ============================================================

    public class TextVectorizer
    {
        private readonly Dictionary<string, int> vocabulary = new();

        private double[] idf;

        private int minNGram = 2;
        private int maxNGram = 4;

        public int VocabularySize => vocabulary.Count;

        public void Fit(List<string> texts)
        {
            var documentFrequency = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var grams = ExtractNGrams(text);

                foreach (var gram in grams.Distinct())
                {
                    if (!documentFrequency.ContainsKey(gram))
                        documentFrequency[gram] = 0;

                    documentFrequency[gram]++;
                }
            }

            int index = 0;

            foreach (var item in documentFrequency
                         .OrderByDescending(x => x.Value))
            {
                vocabulary[item.Key] = index++;
            }

            int documentCount = texts.Count;

            idf = new double[vocabulary.Count];

            foreach (var item in vocabulary)
            {
                int df = documentFrequency[item.Key];

                // Smoothed IDF
                idf[item.Value] =
                    Math.Log((documentCount + 1.0) / (df + 1.0)) + 1.0;
            }
        }

        public double[] Transform(string text)
        {
            double[] vector = new double[vocabulary.Count];

            var grams = ExtractNGrams(text);

            var counts = new Dictionary<string, int>();

            foreach (var gram in grams)
            {
                if (!counts.ContainsKey(gram))
                    counts[gram] = 0;

                counts[gram]++;
            }

            int total = grams.Count;

            if (total == 0)
                return vector;

            foreach (var item in counts)
            {
                if (!vocabulary.TryGetValue(item.Key, out int index))
                    continue;

                // TF
                double tf = (double)item.Value / total;

                // TF-IDF
                vector[index] = tf * idf[index];
            }

            // L2 normalization
            double norm = Math.Sqrt(
                vector.Sum(x => x * x)
            );

            if (norm > 0)
            {
                for (int i = 0; i < vector.Length; i++)
                    vector[i] /= norm;
            }

            return vector;
        }

        private List<string> ExtractNGrams(string text)
        {
            text = Normalize(text);

            var result = new List<string>();

            // 문자 n-gram
            for (int n = minNGram; n <= maxNGram; n++)
            {
                for (int i = 0; i <= text.Length - n; i++)
                {
                    result.Add(text.Substring(i, n));
                }
            }

            return result;
        }

        private string Normalize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return "";

            text = text.Trim().ToLowerInvariant();

            // 불필요한 구두점 제거
            var sb = new StringBuilder();

            foreach (char c in text)
            {
                if (char.IsLetterOrDigit(c) ||
                    c >= 0xAC00 && c <= 0xD7A3 ||
                    c == ' ')
                {
                    sb.Append(c);
                }
            }

            // 여러 공백 -> 하나
            return string.Join(
                " ",
                sb.ToString()
                 .Split(
                     new[] { ' ' },
                     StringSplitOptions.RemoveEmptyEntries
                 )
            );
        }
    }

    // ============================================================
    // 3. 분류 결과
    // ============================================================

    public class Prediction
    {
        public EscapeAction Action { get; set; }
        public double Probability { get; set; }

        public Prediction(EscapeAction action, double probability)
        {
            Action = action;
            Probability = probability;
        }

        public override string ToString()
        {
            return $"=> {Action} " +
                $"({Probability:P1})";
        }
    }

    // ============================================================
    // 4. Multiclass Logistic Regression
    //
    // Softmax:
    //
    // P(class) =
    // exp(score) / sum(exp(score))
    //
    // Gradient Descent로 학습
    // ============================================================

    public class SoftmaxClassifier
    {
        private double[,] weights;
        private double[] bias;

        private List<EscapeAction> labels;

        private int classCount;
        private int featureCount;

        public IReadOnlyList<EscapeAction> Labels => labels;

        public void Fit(
            double[][] x,
            int[] y,
            List<EscapeAction> classLabels,
            int epochs = 1000,
            double learningRate = 0.05)
        {
            labels = classLabels;

            classCount = labels.Count;
            featureCount = x[0].Length;

            weights = new double[classCount, featureCount];
            bias = new double[classCount];

            var random = new System.Random(42);

            // 작은 랜덤값으로 초기화
            for (int c = 0; c < classCount; c++)
            {
                for (int f = 0; f < featureCount; f++)
                {
                    weights[c, f] =
                        (random.NextDouble() - 0.5) * 0.01;
                }
            }

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[,] gradient =
                    new double[classCount, featureCount];

                double[] biasGradient =
                    new double[classCount];

                for (int sample = 0; sample < x.Length; sample++)
                {
                    double[] probabilities =
                        PredictProbabilities(x[sample]);

                    for (int c = 0; c < classCount; c++)
                    {
                        double error =
                            probabilities[c] -
                            (y[sample] == c ? 1.0 : 0.0);

                        for (int f = 0; f < featureCount; f++)
                        {
                            gradient[c, f] +=
                                error * x[sample][f];
                        }

                        biasGradient[c] += error;
                    }
                }

                // Gradient Descent
                for (int c = 0; c < classCount; c++)
                {
                    for (int f = 0; f < featureCount; f++)
                    {
                        weights[c, f] -=
                            learningRate *
                            gradient[c, f] /
                            x.Length;
                    }

                    bias[c] -=
                        learningRate *
                        biasGradient[c] /
                        x.Length;
                }
            }
        }

        public List<Prediction> Predict(
            double[] x,
            int topN = 3)
        {
            double[] probabilities =
                PredictProbabilities(x);

            return Enumerable.Range(0, classCount)
                .Select(i =>
                    new Prediction(
                        labels[i],
                        probabilities[i]))
                .OrderByDescending(x => x.Probability)
                .Take(topN)
                .ToList();
        }

        private double[] PredictProbabilities(double[] x)
        {
            double[] scores =
                new double[classCount];

            for (int c = 0; c < classCount; c++)
            {
                double score = bias[c];

                for (int f = 0; f < featureCount; f++)
                {
                    score += weights[c, f] * x[f];
                }

                scores[c] = score;
            }

            return Softmax(scores);
        }

        private double[] Softmax(double[] scores)
        {
            double max =
                scores.Max();

            double[] exp =
                new double[scores.Length];

            double sum = 0;

            for (int i = 0; i < scores.Length; i++)
            {
                exp[i] =
                    Math.Exp(scores[i] - max);

                sum += exp[i];
            }

            for (int i = 0; i < exp.Length; i++)
                exp[i] /= sum;

            return exp;
        }
    }

    // ============================================================
    // 5. Natural Language -> Action
    // ============================================================

    public class ActionClassifier
    {
        private readonly TextVectorizer vectorizer =
            new TextVectorizer();

        private readonly SoftmaxClassifier classifier =
            new SoftmaxClassifier();

        private readonly List<EscapeAction> actions = new();

        public void Train(
            List<TrainingSample> samples,
            int epochs = 1000,
            double learningRate = 0.05)
        {
            if (samples == null || samples.Count == 0)
                throw new ArgumentException(
                    "학습 데이터가 없습니다.");

            // Action 목록
            actions.AddRange(
                samples
                    .Select(x => x.Action)
                    .Distinct()
                    .OrderBy(x => x)
            );

            // Vectorizer 학습
            vectorizer.Fit(
                samples
                    .Select(x => x.Text)
                    .ToList()
            );

            // 입력 벡터
            double[][] x =
                samples
                    .Select(s =>
                        vectorizer.Transform(s.Text))
                    .ToArray();

            // Action -> 숫자
            int[] y =
                samples
                    .Select(s =>
                        actions.IndexOf(s.Action))
                    .ToArray();

            classifier.Fit(
                x,
                y,
                actions,
                epochs,
                learningRate
            );
        }

        public List<Prediction> Predict(
            string text,
            int topN = 3)
        {
            double[] vector =
                vectorizer.Transform(text);

            return classifier
                .Predict(vector, topN);
        }

        public Prediction PredictAction(
            string text,
            double threshold = 0.50)
        {
            var result =
                Predict(text, 1).First();

            if (result.Probability < threshold)
            {
                return new Prediction(
                    EscapeAction.NoAction,
                    result.Probability);
            }

            return result;
        }
    }

    // ============================================================
    // 6. 실행
    // ============================================================

    public class LBA : MonoBehaviour, ITrain
    {
        //[Header("JSON")] public TextAsset actionJson;
        //private ActionDatabase trainingSamples;
        private readonly ActionClassifier model = new();
        [SerializeField] private UsedModel usedModel;

        private void Awake()
        {
            usedModel.Model = this;
        }

        public void Train(ActionDatabase database)
        {
            // ------------------------------------------------
            // 분류기 생성 및 학습
            // ------------------------------------------------

            UnityEngine.Debug.Log(
                "Natural Language -> Action Classifier"
            );

            List<TrainingSample> samples = new();
            foreach (var sample in database.actions)
            {
                if (sample.action_id != EscapeAction.NoAction)
                {
                    foreach (string text in sample.positive)
                    {
                        samples.Add(new TrainingSample(text, sample.action_id));
                    }
                }
            }

            model.Train(
                samples,
                epochs: 750,
                learningRate: 0.2
            );

            UnityEngine.Debug.Log(
                "학습 완료\n"
            );
        }

        public (EscapeAction, string[], double) Predict(string input, int top = 5)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = model.Predict(input, top);
            stopwatch.Stop();
            return (result[0].Action, result.Select(x => x.ToString()).ToArray(), stopwatch.Elapsed.TotalMilliseconds);
        }
    }
}
