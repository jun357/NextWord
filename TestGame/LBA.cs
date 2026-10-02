using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NaturalLanguageActionClassifier
{
    // ============================================================
    // 1. 학습 데이터
    // ============================================================

    public class TrainingSample
    {
        public string Text { get; }
        public string Action { get; }

        public TrainingSample(string text, string action)
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
        private readonly Dictionary<string, int> vocabulary
            = new Dictionary<string, int>();

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
        public string Action { get; set; }
        public double Probability { get; set; }

        public Prediction(string action, double probability)
        {
            Action = action;
            Probability = probability;
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

        private List<string> labels;

        private int classCount;
        private int featureCount;

        public IReadOnlyList<string> Labels => labels;

        public void Fit(
            double[][] x,
            int[] y,
            List<string> classLabels,
            int epochs = 1000,
            double learningRate = 0.05)
        {
            labels = classLabels;

            classCount = labels.Count;
            featureCount = x[0].Length;

            weights = new double[classCount, featureCount];
            bias = new double[classCount];

            var random = new Random(42);

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

        private readonly List<string> actions =
            new List<string>();

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
                    "UNKNOWN",
                    result.Probability);
            }

            return result;
        }
    }

    // ============================================================
    // 6. 실행
    // ============================================================

    class Program
    {
        static void Main()
        {
            var samples = new List<TrainingSample>
            {
                // -------------------------
                // LIGHT_ON
                // -------------------------

                new TrainingSample(
                    "불 켜줘",
                    "LIGHT_ON"),

                new TrainingSample(
                    "불 좀 켜",
                    "LIGHT_ON"),

                new TrainingSample(
                    "전등 켜줘",
                    "LIGHT_ON"),

                new TrainingSample(
                    "거실 불 켜",
                    "LIGHT_ON"),

                new TrainingSample(
                    "방 불 켜줘",
                    "LIGHT_ON"),

                new TrainingSample(
                    "불을 켜주세요",
                    "LIGHT_ON"),

                // -------------------------
                // LIGHT_OFF
                // -------------------------

                new TrainingSample(
                    "불 꺼줘",
                    "LIGHT_OFF"),

                new TrainingSample(
                    "불 좀 꺼",
                    "LIGHT_OFF"),

                new TrainingSample(
                    "전등 꺼줘",
                    "LIGHT_OFF"),

                new TrainingSample(
                    "거실 불 꺼",
                    "LIGHT_OFF"),

                new TrainingSample(
                    "방 불 꺼줘",
                    "LIGHT_OFF"),

                new TrainingSample(
                    "불을 꺼주세요",
                    "LIGHT_OFF"),

                // -------------------------
                // AC_SET_TEMP
                // -------------------------

                new TrainingSample(
                    "에어컨 24도로 설정해",
                    "AC_SET_TEMP"),

                new TrainingSample(
                    "에어컨 온도 24도",
                    "AC_SET_TEMP"),

                new TrainingSample(
                    "온도를 24도로 해줘",
                    "AC_SET_TEMP"),

                new TrainingSample(
                    "에어컨을 26도로 맞춰",
                    "AC_SET_TEMP"),

                new TrainingSample(
                    "실내 온도 23도로 설정",
                    "AC_SET_TEMP"),

                // -------------------------
                // WEATHER
                // -------------------------

                new TrainingSample(
                    "오늘 날씨 알려줘",
                    "WEATHER"),

                new TrainingSample(
                    "오늘 날씨 어때",
                    "WEATHER"),

                new TrainingSample(
                    "날씨 알려줘",
                    "WEATHER"),

                new TrainingSample(
                    "비 와?",
                    "WEATHER"),

                new TrainingSample(
                    "오늘 비가 오나요",
                    "WEATHER"),

                // -------------------------
                // MUSIC
                // -------------------------

                new TrainingSample(
                    "음악 틀어줘",
                    "PLAY_MUSIC"),

                new TrainingSample(
                    "노래 틀어줘",
                    "PLAY_MUSIC"),

                new TrainingSample(
                    "음악 재생해",
                    "PLAY_MUSIC"),

                new TrainingSample(
                    "노래 재생",
                    "PLAY_MUSIC"),

                new TrainingSample(
                    "음악 좀 틀어",
                    "PLAY_MUSIC")
            };

            // ------------------------------------------------
            // 분류기 생성 및 학습
            // ------------------------------------------------

            var model =
                new ActionClassifier();

            model.Train(
                samples,
                epochs: 1500,
                learningRate: 0.1
            );

            Console.WriteLine(
                "Natural Language -> Action Classifier"
            );

            Console.WriteLine(
                "학습 완료\n"
            );

            // ------------------------------------------------
            // 테스트
            // ------------------------------------------------

            string[] testInputs =
            {
                "거실 전등 좀 켜줄래?",
                "방 불 꺼줘",
                "에어컨 온도 25도로 맞춰줘",
                "오늘 비가 오나요?",
                "노래 하나 틀어줘",
                "안녕하세요"
            };

            foreach (string input in testInputs)
            {
                var result =
                    model.PredictAction(
                        input,
                        threshold: 0.50
                    );

                Console.WriteLine(
                    $"입력 : {input}"
                );

                Console.WriteLine(
                    $"Action : {result.Action}"
                );

                Console.WriteLine(
                    $"Confidence : {result.Probability:P2}"
                );

                Console.WriteLine();

                // Top 3 출력
                Console.WriteLine("Top 3:");

                foreach (var p in
                    model.Predict(input, 3))
                {
                    Console.WriteLine(
                        $"  {p.Action,-15} " +
                        $"{p.Probability:P2}"
                    );
                }

                Console.WriteLine(
                    "----------------------------"
                );
            }

            // ------------------------------------------------
            // 대화형 테스트
            // ------------------------------------------------

            while (true)
            {
                Console.Write(
                    "\n명령 입력 (exit 종료): "
                );

                string input =
                    Console.ReadLine();

                if (string.Equals(
                    input,
                    "exit",
                    StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }

                if (string.IsNullOrWhiteSpace(input))
                    continue;

                var result =
                    model.PredictAction(
                        input,
                        threshold: 0.50
                    );

                Console.WriteLine(
                    $"=> {result.Action} " +
                    $"({result.Probability:P1})"
                );
            }
        }
    }
}
