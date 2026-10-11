using System.Collections.Generic;
using UnityEngine;

public class WeightOptimizer : MonoBehaviour
{
    [SerializeField] private StatisticalMatcher matcher;
    
    // 테스트용 데이터 구조 (예시)
    

    [SerializeField] private List<EvalSample> validationDataset;

    [ContextMenu("Run Random Search Optimization")]
    public void OptimizeWeights()
    {
        float bestLoss = float.MaxValue;
        float bestIntend = 1.5f;
        float bestTarget = 1.5f;
        float bestChar = 1.0f;
        float bestWord = 1.0f;

        int iterations = 50; // 시도해볼 조합 수

        for (int i = 0; i < iterations; i++)
        {
            // 1. 무작위 가중치 생성 (예: 0.1 ~ 3.0 범위)
            float randIntend = UnityEngine.Random.Range(0.2f, 3.0f);
            float randTarget = UnityEngine.Random.Range(0.2f, 3.0f);
            float randChar = UnityEngine.Random.Range(0.2f, 3.0f);
            float randWord = UnityEngine.Random.Range(0.2f, 3.0f);

            // 2. 매처에 가중치 적용
            matcher.IntendWeight = randIntend;
            matcher.TargetWeight = randTarget;
            matcher.CharacterWeight = randChar;
            matcher.WordWeight = randWord;

            // 3. 현재 가중치로 Loss(에러율) 계산
            float currentLoss = EvaluateLoss();

            // 4. 최적값 갱신
            if (currentLoss < bestLoss)
            {
                bestLoss = currentLoss;
                bestIntend = randIntend;
                bestTarget = randTarget;
                bestChar = randChar;
                bestWord = randWord;
                
                Debug.Log($"[New Best] Loss: {bestLoss:F3} | Intend: {bestIntend:F2}, Target: {bestTarget:F2}, Char: {bestChar:F2}, Word: {bestWord:F2}");
            }
        }

        // 최종 최적 가중치 확정 적용
        matcher.IntendWeight = bestIntend;
        matcher.TargetWeight = bestTarget;
        matcher.CharacterWeight = bestChar;
        matcher.WordWeight = bestWord;
        
        Debug.Log("Optimization Finished!");
    }

    private float EvaluateLoss()
    {
        if (validationDataset == null || validationDataset.Count == 0) return 0f;

        int errorCount = 0;

        foreach (var sample in validationDataset)
        {
            // Match 메서드 호출 (Top 1 결과 확인)
            var results = matcher.Match(sample.input, 1);

            if (results.Count == 0 || results[0].ActionId != sample.expectedActionId)
            {
                errorCount++;
            }
        }

        // 에러율 반환 (0.0 ~ 1.0, 낮을수록 좋음)
        return (float)errorCount / validationDataset.Count;
    }
}