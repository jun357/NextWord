using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using UnityEngine;
using System.Collections.Concurrent;
using System.Threading.Tasks;

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

    // ============================================================
    // 5. Natural Language -> Action
    // ============================================================

    // ============================================================
    // 6. 실행
    // ============================================================

    public class LBA : MonoBehaviour, ITrain
    {
        [Header("JSON")] public TextAsset pre_trained;
        private ActionClassifier model = new(new(), new(), new());
        [SerializeField] private UsedModel usedModel;

        private void Start()
        {
            if (usedModel != null)
                usedModel.Model = this;
        }

        public void Train(ActionDatabase database)
        {
            if (pre_trained != null)
            {
                UnityEngine.Debug.Log("model found");
                model = JsonConvert.DeserializeObject<ActionClassifier>(pre_trained.text);
                return;
            }

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
                validationRatio: 0.2,
                seed: 42,
                maxEpochs: 100,
                maxParallelism: 2
            );

            UnityEngine.Debug.Log(
                "학습 완료\n"
            );

            SaveModel();
        }

        public (EscapeAction, string[], double) Predict(string input, int top = 5)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = model.Predict(input, top);
            stopwatch.Stop();
            return (result[0].Action, result.Select(x => x.ToString()).ToArray(), stopwatch.Elapsed.TotalMilliseconds);
        }

        private void SaveModel()
        {
            string date = DateTime.Now.ToString("yyyy-MM-dd");

            string path = System.IO.Path.Combine(
                Application.persistentDataPath,
                $"ActionModel_{date}.json"
            );

            string json = JsonConvert.SerializeObject(
                model,
                Formatting.Indented
            );

            System.IO.File.WriteAllText(path, json);

            UnityEngine.Debug.Log(
                $"모델 저장 완료: {path}"
            );
        }
    }
}