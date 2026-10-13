using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

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

        public bool WasPreTrained => pre_trained != null;

        private void Start()
        {
            if (usedModel != null)
                usedModel.Model = this;
        }

        public void AutoFit(List<TrainingSample> trainSamples, List<TrainingSample> validationSamples, List<string> ids, Dictionary<string, int> dict)
        {
            // ------------------------------------------------
            // 분류기 생성 및 학습
            // ------------------------------------------------

            UnityEngine.Debug.Log(
                "Natural Language -> Action Classifier"
            );

            model.Train(
                trainSamples,
                validationSamples,
                ids,
                dict,
                seed: 42,
                maxEpochs: 1,
                maxParallelism: 3
            );

            UnityEngine.Debug.Log(
                "학습 완료\n"
            );

            SaveModel();
        }

        public (string, string[], double) Predict(string input, int top = 5)
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
                $"LinearModel_{date}.json"
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

        public void Train(ActionDatabase database)
        {
            throw new NotImplementedException();
        }

        public void LoadModel(ActionDatabase _)
        {
            model = JsonConvert.DeserializeObject<ActionClassifier>(pre_trained.text);
            return;
        }
    }
}