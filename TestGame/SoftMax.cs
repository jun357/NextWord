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
        [Newtonsoft.Json.JsonProperty] private List<string> labels;

        [Newtonsoft.Json.JsonProperty] private int classCount;
        [Newtonsoft.Json.JsonProperty] private int featureCount;

        [Newtonsoft.Json.JsonIgnore] public IReadOnlyList<string> Labels => labels;

        public (double accuracy, double loss) Evaluate(
    SparseVector[] x,
    int[] y)
        {
            if (x == null || x.Length == 0)
                return (0, 0);

            if (y == null || y.Length != x.Length)
                throw new ArgumentException(
                    "평가 데이터와 레이블 길이가 다릅니다.");

            int correct = 0;
            double loss = 0;

            double[] probabilityBuffer = new double[classCount];

            for (int i = 0; i < x.Length; i++)
            {
                PredictProbabilities(x[i], probabilityBuffer);

                int predicted = ArgMax(probabilityBuffer);
                int target = y[i];

                if (predicted == target)
                    correct++;

                loss -= Math.Log(
                    Math.Max(1e-15, probabilityBuffer[target]));
            }

            return (
                (double)correct / x.Length,
                loss / x.Length
            );
        }

        public void Fit(
            SparseVector[] trainX,
            int[] trainY,
            SparseVector[] valX,
            int[] valY,
            List<string> classLabels,
            double learningRate,
            int maxEpochs,
            int seed,
            Action<int, double, double> onEpoch = null,
            int evalStep = 50,
            int patience = 5,
            double minDelta = 0.0001
        )
        {
            if (trainX == null || trainX.Length == 0)
                throw new ArgumentException("Train 데이터가 없습니다.");

            if (trainY == null || trainY.Length != trainX.Length)
                throw new ArgumentException(
                    "Train 데이터와 레이블 길이가 다릅니다.");

            if (evalStep <= 0)
                throw new ArgumentOutOfRangeException(nameof(evalStep));

            if (maxEpochs <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxEpochs));

            if (patience <= 0)
                throw new ArgumentOutOfRangeException(nameof(patience));

            if (minDelta < 0 || double.IsNaN(minDelta))
                throw new ArgumentOutOfRangeException(nameof(minDelta));

            if (classLabels == null || classLabels.Count == 0)
                throw new ArgumentException("클래스가 없습니다.");

            labels = new List<string>(classLabels);
            classCount = labels.Count;

            if (featureCount <= 0)
                throw new InvalidOperationException(
                    "Feature 수가 초기화되지 않았습니다.");

            if (valX != null && valY == null)
                throw new ArgumentException("Validation 레이블이 없습니다.");

            if (valX != null && valX.Length != valY.Length)
                throw new ArgumentException(
                    "Validation 데이터와 레이블 길이가 다릅니다.");

            weights = new double[classCount, featureCount];
            bias = new double[classCount];

            double[] probabilityBuffer = new double[classCount];

            var random = new Random(seed);
            var order = Enumerable.Range(0, trainX.Length).ToArray();

            double bestLoss = double.PositiveInfinity;
            double[,] bestWeights = null;
            double[] bestBias = null;

            int bestEpoch = 0;
            int evaluationsWithoutImprovement = 0;

            const double l2 = 0.0001;

            bool hasValidation = valX != null && valX.Length > 0;

            for (int epoch = 1; epoch <= maxEpochs; epoch++)
            {
                // Fisher-Yates shuffle
                for (int i = order.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }

                foreach (int sample in order)
                {
                    SparseVector vector = trainX[sample];

                    PredictProbabilities(vector, probabilityBuffer);

                    int target = trainY[sample];

                    if ((uint)target >= (uint)classCount)
                        throw new ArgumentOutOfRangeException(
                            nameof(trainY),
                            "Train 레이블 인덱스가 클래스 범위를 벗어났습니다.");

                    for (int c = 0; c < classCount; c++)
                    {
                        double error =
                            probabilityBuffer[c] -
                            (c == target ? 1.0 : 0.0);

                        bias[c] -= learningRate * error;

                        for (int j = 0; j < vector.Indices.Length; j++)
                        {
                            int f = vector.Indices[j];

                            if ((uint)f >= (uint)featureCount)
                                continue;

                            weights[c, f] -= learningRate * (error * vector.Values[j] + l2 * weights[c, f]);
                        }
                    }
                }

                // Validation이 없는 경우 조기 종료를 적용하지 않음
                if (!hasValidation)
                    continue;

                // evalStep마다 평가하고 마지막 Epoch에서도 평가
                if (epoch % evalStep != 0 && epoch != maxEpochs)
                    continue;

                var metrics = Evaluate(valX, valY);

                onEpoch?.Invoke(
                    epoch,
                    metrics.accuracy,
                    metrics.loss);

                // minDelta 이상 개선된 경우에만 Best 모델 갱신
                if (metrics.loss < bestLoss - minDelta)
                {
                    bestLoss = metrics.loss;
                    bestEpoch = epoch;

                    bestWeights = (double[,])weights.Clone();
                    bestBias = (double[])bias.Clone();

                    evaluationsWithoutImprovement = 0;
                }
                else
                {
                    evaluationsWithoutImprovement++;
                }

                // 일정 횟수 동안 개선되지 않으면 학습 종료
                if (evaluationsWithoutImprovement >= patience)
                {
                    break;
                }
            }

            // Validation Loss가 가장 낮았던 모델 복원
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
                .Select(i => new Prediction(labels[i], probabilities[i]))
                .OrderByDescending(p => p.Probability)
                .Take(topN)
                .ToList();
        }

        private double[] PredictProbabilities(SparseVector x)
        {
            var scores = new double[classCount];
            PredictProbabilities(x, scores);
            return scores;
        }

        private void PredictProbabilities(
            SparseVector x,
            double[] scores)
        {
            double max = double.NegativeInfinity;

            for (int c = 0; c < classCount; c++)
            {
                double score = bias[c];

                for (int j = 0; j < x.Indices.Length; j++)
                {
                    int f = x.Indices[j];

                    if ((uint)f < (uint)featureCount)
                        score += weights[c, f] * x.Values[j];
                }

                scores[c] = score;

                if (score > max)
                    max = score;
            }

            double sum = 0;

            for (int c = 0; c < classCount; c++)
            {
                double value = Math.Exp(scores[c] - max);
                scores[c] = value;
                sum += value;
            }

            double invSum = 1.0 / sum;

            for (int c = 0; c < classCount; c++)
                scores[c] *= invSum;
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