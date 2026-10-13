using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace NaturalLanguageActionClassifier
{
    [Serializable]
    public class ActionClassifier
    {
        private readonly TextVectorizer vectorizer;
        public TextVectorizer textVectorizer => vectorizer;
        private SoftmaxClassifier classifier;
        public SoftmaxClassifier softmaxClassifier => classifier;

        private List<string> actions;

        public IReadOnlyList<string> Actions => actions;
        //public int VocabularySize => vectorizer.VocabularySize;

        public double BestLearningRate { get; private set; }
        public int BestEpoch { get; private set; }
        public double BestValidationAccuracy { get; private set; }
        public double BestValidationLoss { get; private set; }

        [JsonConstructor]
        public ActionClassifier(
            TextVectorizer textVectorizer,
            SoftmaxClassifier softmaxClassifier,
            List<string> Actions)
        {
            vectorizer = textVectorizer
                ?? throw new ArgumentNullException(nameof(vectorizer));

            classifier = softmaxClassifier
                ?? throw new ArgumentNullException(nameof(classifier));

            actions = Actions 
                ?? throw new ArgumentNullException(nameof(classifier));
        }

        public void Train(
            List<TrainingSample> trainSamples,
            List<TrainingSample> validationSamples,
            List<string> actions,
            Dictionary<string, int> labelToIndex,
            int seed,
            int maxEpochs,
            int maxParallelism)
        {
            if (trainSamples == null || trainSamples.Count == 0)
                throw new ArgumentException("학습 데이터가 없습니다.");

            if (validationSamples == null || validationSamples.Count == 0)
                throw new InvalidOperationException("Validation 데이터가 없습니다.");

            if (trainSamples.Any(s => s == null || string.IsNullOrWhiteSpace(s.Text))
                || validationSamples.Any(s => s == null || string.IsNullOrWhiteSpace(s.Text)))
            {
                throw new ArgumentException("빈 문장 또는 null 학습 데이터가 있습니다.");
            }

            // Action 목록 초기화
            this.actions = actions;

            if (actions.Count == 0)
                throw new InvalidOperationException("학습 가능한 Action이 없습니다.");

            // Vocabulary는 Train 데이터만 사용해 학습
            vectorizer.Fit(
                trainSamples.Select(s => s.Text).ToList());

            // Sparse TF-IDF 변환
            SparseVector[] trainX = trainSamples
                .Select(s => vectorizer.TransformSparse(s.Text))
                .ToArray();

            SparseVector[] validationX = validationSamples
                .Select(s => vectorizer.TransformSparse(s.Text))
                .ToArray();

            int[] trainY = trainSamples
                .Select(s => labelToIndex[s.Action])
                .ToArray();

            int[] validationY = validationSamples
                .Select(s => labelToIndex[s.Action])
                .ToArray();

            Debug.Log(
                $"[Data] Train={trainX.Length}, " +
                $"Validation={validationX.Length}, " +
                $"Actions={actions.Count}, " +
                $"Features={vectorizer.VocabularySize}");

            // 각 LR 후보는 독립적인 모델을 사용
            double[] learningRates =
            {
                //0.01, 0.03, 0.1, 0.3
                0.4
            };

            var results = new System.Collections.Concurrent
                .ConcurrentBag<TuningResult>();

            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = Math.Max(
                    1,
                    Math.Min(maxParallelism,
                        Environment.ProcessorCount))
            };

            Parallel.ForEach(
            learningRates,
            options,
            learningRate =>
            {
                var candidate = new SoftmaxClassifier();

                candidate.InitializeFeatureCount(
                    vectorizer.VocabularySize);

                candidate.Fit(
                    trainX,
                    trainY,
                    validationX,
                    validationY,
                    actions,
                    learningRate,
                    maxEpochs,
                    seed,
                    onEpoch: (epoch, accuracy, loss) =>
                    {
                        // 중간 평가 로그는 필요할 때만 출력
                        if (epoch % 100 == 0 || epoch == maxEpochs)
                        {
                            lock (Console.Out)
                            {
                                Debug.Log(
                                $"[Tune] LR={learningRate:F3}, " +
                                $"Epoch={epoch}, " +
                                $"ValAcc={accuracy:P2}, " +
                                $"ValLoss={loss:F4}");
                            }
                        }
                    },
                    evalStep: 10,
                    patience: 5,
                    minDelta: 0.0001);

                var metrics = candidate.Evaluate(validationX, validationY);

                results.Add(new TuningResult
                {
                    LearningRate = learningRate,
                    Epochs = maxEpochs,
                    Accuracy = metrics.accuracy,
                    Loss = metrics.loss,
                    Model = candidate
                });
            });

            // Accuracy 우선, 동률이면 Loss가 낮은 모델 선택
            var best = results
            .OrderByDescending(r => r.Accuracy)
            .ThenBy(r => r.Loss)
            .First();

            classifier = best.Model;

            BestLearningRate = best.LearningRate;
            BestEpoch = best.Epochs;
            BestValidationAccuracy = best.Accuracy;
            BestValidationLoss = best.Loss;

            Debug.Log(
            $"[Best Model] LR={BestLearningRate:F3}, " +
            $"Epoch={BestEpoch}, " +
            $"ValAcc={BestValidationAccuracy:P2}, " +
            $"ValLoss={BestValidationLoss:F4}");

            double avg = trainX.Average(x => x.Indices.Length);
            int max = trainX.Max(x => x.Indices.Length);

            Debug.Log(
                $"[Sparse] Vocab={vectorizer.VocabularySize}, " +
                $"AvgNonZero={avg:F1}, MaxNonZero={max}");
        }

        public List<Prediction> Predict(
            string text,
            int topN = 3)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new List<Prediction>
                {
                    new Prediction(Normalizer.NULL_ID, 1.0)
                };
            }

            if (actions.Count == 0)
                throw new InvalidOperationException(
                    "모델을 먼저 학습해야 합니다.");

            SparseVector vector =
                vectorizer.TransformSparse(text);

            return classifier.Predict(
                vector,
                Math.Max(1, Math.Min(topN, actions.Count)));
        }

        public Prediction PredictAction(
            string text,
            double threshold = 0.5)
        {
            var predictions = Predict(text, 1);
            var best = predictions[0];

            if (best.Probability < threshold)
            {
                return new Prediction(
                    Normalizer.NULL_ID,
                    best.Probability);
            }

            return best;
        }

        private sealed class TuningResult
        {
            public double LearningRate;
            public int Epochs;
            public double Accuracy;
            public double Loss;
            public SoftmaxClassifier Model;
        }
    }
}