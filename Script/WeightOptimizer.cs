using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;

public class WeightOptimizer : MonoBehaviour
{
    [SerializeField] private StatisticalMatcher matcher;
   
    public void OptimizeWeights(List<TrainingSample> validationDataset)
    {
        const int iterations = 8;

        // 1. 메인 스레드에서 가중치 조합 미리 생성
        var weights = new StatisticalWeight[iterations];

        for (int i = 0; i < iterations; i++)
        {
            weights[i] = new StatisticalWeight(
                UnityEngine.Random.Range(0.2f, 3.0f),
                UnityEngine.Random.Range(0.2f, 3.0f),
                UnityEngine.Random.Range(0.2f, 3.0f),
                UnityEngine.Random.Range(0.2f, 3.0f),
                UnityEngine.Random.Range(0.2f, 3.0f)
            );
        }

        // 2. CPU 코어 수를 고려해 작업 분배
        int workerCount = Math.Min(
            Environment.ProcessorCount,
            iterations
        );

        var localBestLoss = new float[workerCount];
        var localBestWeights = new StatisticalWeight[workerCount];

        for (int i = 0; i < workerCount; i++)
        {
            localBestLoss[i] = float.MaxValue;
            localBestWeights[i] = new StatisticalWeight(1f, 1f, 1f, 1f, 1f);
        }

        // 3. 각 작업자가 독립적으로 최적값 탐색
        Parallel.For(0, workerCount, worker =>
        {
            float bestLoss = float.MaxValue;
            StatisticalWeight bestWeight = new StatisticalWeight(1f, 1f, 1f, 1f, 1f);

            // 각 작업자가 담당할 구간
            int start = worker * iterations / workerCount;
            int end = (worker + 1) * iterations / workerCount;

            for (int i = start; i < end; i++)
            {
                StatisticalWeight weight = weights[i];

                float loss = EvaluateLoss(validationDataset, weight);

                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    bestWeight = weight;
                }
            }

            // 작업자마다 다른 배열 인덱스에 기록하므로 lock 불필요
            localBestLoss[worker] = bestLoss;
            localBestWeights[worker] = bestWeight;
        });

        // 4. 모든 작업이 끝난 후 최적값 통합
        float globalBestLoss = float.MaxValue;
        StatisticalWeight globalBestWeight = new StatisticalWeight(1f, 1f, 1f, 1f, 1f);

        for (int i = 0; i < workerCount; i++)
        {
            if (localBestLoss[i] < globalBestLoss)
            {
                globalBestLoss = localBestLoss[i];
                globalBestWeight = localBestWeights[i];
            }
        }

        // 5. 최종 결과 적용
        SaveModel(globalBestWeight);

        Debug.Log(
            $"Optimization Finished! " +
            $"Best Loss: {globalBestLoss:F3} | " +
            $"Intend: {globalBestWeight.intendWeight:F2}, " +
            $"Target: {globalBestWeight.targetWeight:F2}, " +
            $"Context: {globalBestWeight.contextWeight:F2}, " +
            $"Char: {globalBestWeight.characterWeight:F2}, " +
            $"Word: {globalBestWeight.wordWeight:F2}"
        );

        /*var globalBestWeight = new StatisticalWeight(1f, 1f, 1f, 1f, 1f);

        float globalBestLoss = EvaluateLoss(validationDataset, globalBestWeight);
        SaveModel(globalBestWeight);

        Debug.Log(
            $"Optimization Finished! " +
            $"Best Loss: {globalBestLoss:F3} | " +
            $"Intend: {globalBestWeight.intendWeight:F2}, " +
            $"Target: {globalBestWeight.targetWeight:F2}, " +
            $"Context: {globalBestWeight.contextWeight:F2}, " +
            $"Char: {globalBestWeight.characterWeight:F2}, " +
            $"Word: {globalBestWeight.wordWeight:F2}"
        );*/
    }

    private float EvaluateLoss(List<TrainingSample> validationDataset, StatisticalWeight weight)
    {
        if (validationDataset == null || validationDataset.Count == 0) return 0f;

        int errorCount = 0;

        foreach (var sample in validationDataset)
        {
            // Match 메서드 호출 (Top 1 결과 확인)
            var results = matcher.Match(sample.Text, weight, 1);

            if (results.Count == 0 || results[0].ActionId != sample.Action)
            {
                errorCount++;
            }
        }

        // 에러율 반환 (0.0 ~ 1.0, 낮을수록 좋음)
        return (float)errorCount / validationDataset.Count;
    }

    private void SaveModel(StatisticalWeight weight)
    {
        string date = DateTime.Now.ToString("yyyy-MM-dd");

        string path = System.IO.Path.Combine(
            Application.persistentDataPath,
            $"StatisticalModel_{date}.json"
        );

        string json = JsonConvert.SerializeObject(
            weight,
            Formatting.Indented
        );

        System.IO.File.WriteAllText(path, json);

        UnityEngine.Debug.Log(
            $"모델 저장 완료: {path}"
        );
    }
}