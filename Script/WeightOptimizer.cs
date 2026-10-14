using NaturalLanguageActionClassifier;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;

public class WeightOptimizer : MonoBehaviour
{
    [SerializeField] private StatisticalMatcher matcher;

    public static Vector2 oneVector2 = new Vector2(1, 1);

    public static Vector2 RandomVector2()
    {
        return new Vector2(UnityEngine.Random.Range(1f, 3.0f), UnityEngine.Random.Range(1f, 3.0f));
    }

    public void OptimizeWeightsPro(List<TrainingSample> validationDataset, List<(Vector2, Vector2, Vector2, Vector2)> weights = null)
    {
        if (weights == null)
        {
            weights = new()
            {
                (oneVector2, oneVector2, oneVector2, oneVector2)
            };
            for (int i = 0; i < 0; i++)
                weights.Add((RandomVector2(), RandomVector2(), RandomVector2(), RandomVector2()));
        }

        var cache = new List<(Vector2[], Dictionary<(SlotFeature, string), Vector2>, string)>();
        foreach (var data in validationDataset)
        {
            (var scores, var featureScore) = matcher.TextToScoreMap(data.Text);
            cache.Add((scores, featureScore, data.Action));
        }

        var best_loss = 1f;
        var best_weight = (new Vector2(), new Vector2(), new Vector2(), new Vector2());
        foreach ((var w1, var w2, var w3, var w4) in weights)
        {
            int errorCount = 0;
            foreach ((var scores, var featureScore, string id) in cache)
            {
                string answer = matcher.ScoreMapToScore(scores, featureScore, w1, w2, w3, w4);
                if (answer != id)
                    errorCount++;
            }

            float loss = (float)errorCount / validationDataset.Count;
            if (loss < best_loss)
            {
                best_loss = loss;
                best_weight = (w1, w2, w3, w4);
            }
        }

        Debug.Log(
            $"Optimization Finished! " +
            $"Best Loss: {best_loss:F3} | " +
            $"Intend: {best_weight.Item1}, " +
            $"Target: {best_weight.Item2}, " +
            $"Context: {best_weight.Item3}, " +
            $"Action: {best_weight.Item4}"
        );

        matcher.weights = best_weight.Flatten();
    }

    public void OptimizeWeights(List<TrainingSample> validationDataset)
    {
        const int iterations = 500;

        // 1. 메인 스레드에서 가중치 조합 미리 생성
        var weights = new Vector2[iterations, 4];

        for (int i = 0; i < iterations; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                weights[i,j] = new (UnityEngine.Random.Range(0.5f, 5.0f), UnityEngine.Random.Range(0.5f, 5.0f));
            }
        }

        // 2. CPU 코어 수를 고려해 작업 분배
        int workerCount = Math.Min(
            Environment.ProcessorCount,
            iterations
        );

        var localBestLoss = new float[workerCount];
        var localBestWeights = new Vector2[workerCount, 4];

        for (int k = 0; k < workerCount; k++)
        {
            localBestLoss[k] = float.MaxValue;
            for (int j = 0; j < 4; j++)
            {
                localBestWeights[k,j] = new(1f, 1f);
            }
        }

        // 3. 각 작업자가 독립적으로 최적값 탐색
        Parallel.For(0, workerCount, worker =>
        {
            float bestLoss = float.MaxValue;
            var bestWeight = new Vector2[4];
            for (int j = 0; j < 4; j++)
            {
                bestWeight[j] = new(1f, 1f);
            }

            // 각 작업자가 담당할 구간
            int start = worker * iterations / workerCount;
            int end = (worker + 1) * iterations / workerCount;

            for (int i = start; i < end; i++)
            {
                Vector2 w1 = weights[worker, 0];
                Vector2 w2 = weights[worker, 1];
                Vector2 w3 = weights[worker, 2];
                Vector2 w4 = weights[worker, 3];

                float loss = EvaluateLoss(validationDataset, w1,w2, w3, w4);

                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    bestWeight = new Vector2[4]
                    {
                        w1, w2, w3, w4
                    };
                }
            }

            // 작업자마다 다른 배열 인덱스에 기록하므로 lock 불필요
            localBestLoss[worker] = bestLoss;
            localBestWeights[worker, 0] = bestWeight[0];
            localBestWeights[worker, 1] = bestWeight[1];
            localBestWeights[worker, 2] = bestWeight[2];
            localBestWeights[worker, 3] = bestWeight[3];
        });

        // 4. 모든 작업이 끝난 후 최적값 통합
        float globalBestLoss = float.MaxValue;
        var globalBestWeight = new Vector2[4];
        for (int j = 0; j < 4; j++)
        {
            globalBestWeight[j] = new(1f, 1f);
        }

        for (int i = 0; i < workerCount; i++)
        {
            if (localBestLoss[i] < globalBestLoss)
            {
                globalBestLoss = localBestLoss[i];
                globalBestWeight[0] = localBestWeights[i, 0];
                globalBestWeight[1] = localBestWeights[i, 1];
                globalBestWeight[2] = localBestWeights[i, 2];
                globalBestWeight[3] = localBestWeights[i, 3];
            }
        }

        // 5. 최종 결과 적용
        SaveModel(globalBestWeight);

        Debug.Log(
            $"Optimization Finished! " +
            $"Best Loss: {globalBestLoss:F3} | " +
            $"Intend: {globalBestWeight[0]}, " +
            $"Target: {globalBestWeight[1]}, " +
            $"Context: {globalBestWeight[2]}, " +
            $"Action: {globalBestWeight[3]}, "
        );
    }

    private float EvaluateLoss(List<TrainingSample> validationDataset, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4)
    {
        if (validationDataset == null || validationDataset.Count == 0) return 0f;

        int errorCount = 0;

        foreach (var sample in validationDataset)
        {
            // Match 메서드 호출 (Top 1 결과 확인)
            var results = matcher.Match(sample.Text, v1, v2, v3, v4, 1);

            if (results.Count == 0 || results[0].ActionId != sample.Action)
            {
                errorCount++;
            }
        }

        // 에러율 반환 (0.0 ~ 1.0, 낮을수록 좋음)
        return (float)errorCount / validationDataset.Count;
    }

    private void SaveModel(Vector2[] weight)
    {
        string date = DateTime.Now.ToString("yyyy-MM-dd");

        string path = System.IO.Path.Combine(
            Application.persistentDataPath,
            $"StatisticalModel_{date}.json"
        );

        string json = JsonConvert.SerializeObject(
            weight.Flatten(),
            Formatting.Indented
        );

        System.IO.File.WriteAllText(path, json);

        UnityEngine.Debug.Log(
            $"모델 저장 완료: {path}"
        );
    }
}

public static class VectorExtensions
{
    public static float[] Flatten(this (Vector2, Vector2, Vector2, Vector2) vectors)
    {
        float[] result = new float[8];
        result[0] = vectors.Item1.x;
        result[1] = vectors.Item1.y;
        result[2] = vectors.Item2.x;
        result[3] = vectors.Item2.y;
        result[4] = vectors.Item3.x;
        result[5] = vectors.Item3.y;
        result[6] = vectors.Item4.x;
        result[7] = vectors.Item4.y;
        return result;
    }

    /// <summary>
    /// Vector2 배열을 x, y 순서의 float 배열로 플래튼(Flatten)합니다.
    /// </summary>
    public static float[] Flatten(this Vector2[] vectors)
    {
        if (vectors == null) return null;

        float[] result = new float[vectors.Length * 2];
        for (int i = 0; i < vectors.Length; i++)
        {
            result[i * 2] = vectors[i].x;
            result[i * 2 + 1] = vectors[i].y;
        }

        return result;
    }

    /// <summary>
    /// float 배열을 x, y 쌍으로 묶어 Vector2 배열로 변환합니다.
    /// </summary>
    public static Vector2[] ToVector2Array(this float[] floats)
    {
        if (floats == null) return null;

        // 데이터가 짝수 개가 아닐 경우 처리 (필요에 따라 예외 처리 방식 변경 가능)
        if (floats.Length % 2 != 0)
        {
            Debug.LogWarning("float 배열의 길이가 짝수가 아닙니다. 마지막 요소는 무시됩니다.");
        }

        int length = floats.Length / 2;
        Vector2[] result = new Vector2[length];

        for (int i = 0; i < length; i++)
        {
            result[i] = new Vector2(floats[i * 2], floats[i * 2 + 1]);
        }

        return result;
    }
}