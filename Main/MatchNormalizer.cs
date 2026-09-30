using System;
using System.Collections.Generic;
using System.Linq;

public class MatchNormalizer
{
    public float NoMatchRatio { get; set; } = 0.30f;

    public float AmbiguousRatio { get; set; } = 0.10f;

    public float MinimumConfidence { get; set; } = 0.35f;

    public float GetRelativeScore(
        float score,
        float maxScore)
    {
        if (maxScore <= 0f)
            return 0f;

        return score / maxScore;
    }

    public float GetRelativeGap(
        float topScore,
        float secondScore)
    {
        if (topScore <= 0f)
            return 0f;

        return
            (topScore - secondScore)
            / topScore;
    }

    public MatchState Classify(
        List<MatchResult> results)
    {
        if (results == null ||
            results.Count == 0)
        {
            return MatchState.NoMatch;
        }

        MatchResult top =
            results[0];

        // ------------------------------------------
        // 슬롯 판정은 score보다 먼저
        // ------------------------------------------

        if (!top.IntendExplicit)
        {
            if (top.TargetExplicit)
                return MatchState.NeedIntend;

            return MatchState.NoMatch;
        }

        if (!top.TargetExplicit)
        {
            return MatchState.NeedTarget;
        }

        // ------------------------------------------
        // 전체 confidence
        // ------------------------------------------

        float maxScore =
            results.Max(x => x.FinalScore);

        float relativeScore =
            GetRelativeScore(
                top.FinalScore,
                maxScore);

        if (relativeScore <
            NoMatchRatio)
        {
            return MatchState.NoMatch;
        }

        // ------------------------------------------
        // 후보간 상대 차이
        // ------------------------------------------

        if (results.Count >= 2)
        {
            float secondScore =
                results[1].FinalScore;

            float gap =
                GetRelativeGap(
                    top.FinalScore,
                    secondScore);

            if (gap <
                AmbiguousRatio)
            {
                return MatchState.Ambiguous;
            }
        }

        return MatchState.Confident;
    }
}