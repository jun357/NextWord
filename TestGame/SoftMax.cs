using System;
using System.Collections.Generic;
using System.Linq;

namespace NaturalLanguageActionClassifier
{
    [Serializable]
    public class SoftmaxClassifier
    {
        [Newtonsoft.Json.JsonProperty] private double[,] weights;
        [Newtonsoft.Json.JsonProperty] private double[] bias;
        [Newtonsoft.Json.JsonProperty] private List<EscapeAction> labels;

        [Newtonsoft.Json.JsonProperty] private int classCount;
        [Newtonsoft.Json.JsonProperty] private int featureCount;

        [Newtonsoft.Json.JsonIgnore] public IReadOnlyList<EscapeAction> Labels => labels;

        public (double accuracy, double loss) Evaluate(
            SparseVector[] x, int[] y)
        {
            int correct = 0;
            double loss = 0;

            for (int i = 0; i < x.Length; i++)
            {
                double[] p = PredictProbabilities(x[i]);

                if (ArgMax(p) == y[i])
                    correct++;

                loss -= Math.Log(
                    Math.Max(1e-15, p[y[i]]));
            }

            return (
                x.Length == 0 ? 0 : (double)correct / x.Length,
                x.Length == 0 ? 0 : loss / x.Length
            );
        }

        public void Fit(
            SparseVector[] trainX,
            int[] trainY,
            SparseVector[] valX,
            int[] valY,
            List<EscapeAction> classLabels,
            double learningRate,
            int maxEpochs,
            int seed,
            Action<int, double, double> onEpoch = null)
        {
            if (trainX == null || trainX.Length == 0)
                throw new ArgumentException("Train 데이터가 없습니다.");

            labels = new List<EscapeAction>(classLabels);
            classCount = labels.Count;

            // 전체 Vocabulary 크기를 외부에서 전달할 수 없으므로
            // 호출 전에 InitializeFeatureCount를 사용합니다.
            if (featureCount <= 0)
                throw new InvalidOperationException(
                    "Feature 수가 초기화되지 않았습니다.");

            weights = new double[classCount, featureCount];
            bias = new double[classCount];

            var random = new Random(seed);
            var order = Enumerable.Range(0, trainX.Length).ToArray();

            // Best validation checkpoint
            double bestLoss = double.PositiveInfinity;
            double[,] bestWeights = null;
            double[] bestBias = null;

            const double l2 = 0.0001;

            for (int epoch = 1; epoch <= maxEpochs; epoch++)
            {
                // 매 Epoch 데이터 순서 섞기
                for (int i = order.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }

                foreach (int sample in order)
                {
                    SparseVector vector = trainX[sample];
                    double[] p = PredictProbabilities(vector);
                    int target = trainY[sample];

                    for (int c = 0; c < classCount; c++)
                    {
                        double error =
                            p[c] - (c == target ? 1.0 : 0.0);

                        bias[c] -= learningRate * error;

                        for (int j = 0;
                             j < vector.Indices.Length;
                             j++)
                        {
                            int f = vector.Indices[j];

                            weights[c, f] -= learningRate *
                                (error * vector.Values[j]
                                 + l2 * weights[c, f]);
                        }
                    }
                }

                // Validation은 각 Epoch 종료 후 실행
                if (valX != null && valX.Length > 0)
                {
                    var metrics = Evaluate(valX, valY);

                    onEpoch?.Invoke(
                        epoch, metrics.accuracy, metrics.loss);

                    if (metrics.loss < bestLoss)
                    {
                        bestLoss = metrics.loss;
                        bestWeights = (double[,])weights.Clone();
                        bestBias = (double[])bias.Clone();
                    }
                }
            }

            if (bestWeights != null)
            {
                weights = bestWeights;
                bias = bestBias;
            }
        }

        // SparseVector의 Indices는 실제 Vocabulary 인덱스이므로
        // 모델 생성 시 전체 Vocabulary 크기를 지정해야 합니다.
        public void InitializeFeatureCount(int count)
        {
            if (count <= 0)
                throw new ArgumentOutOfRangeException(nameof(count));

            featureCount = count;
        }

        public List<Prediction> Predict(
            SparseVector x, int topN = 3)
        {
            var probabilities = PredictProbabilities(x);

            return Enumerable.Range(0, classCount)
                .Select(i => new Prediction(
                    labels[i], probabilities[i]))
                .OrderByDescending(p => p.Probability)
                .Take(topN)
                .ToList();
        }

        private double[] PredictProbabilities(SparseVector x)
        {
            double[] scores = new double[classCount];

            for (int c = 0; c < classCount; c++)
            {
                double score = bias[c];

                for (int j = 0; j < x.Indices.Length; j++)
                {
                    int f = x.Indices[j];

                    if (f < featureCount)
                        score += weights[c, f] * x.Values[j];
                }

                scores[c] = score;
            }

            double max = scores.Max();
            double sum = 0;

            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] = Math.Exp(scores[i] - max);
                sum += scores[i];
            }

            for (int i = 0; i < scores.Length; i++)
                scores[i] /= sum;

            return scores;
        }

        private static int ArgMax(double[] values)
        {
            int best = 0;

            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[best])
                    best = i;
            }

            return best;
        }
    }
}