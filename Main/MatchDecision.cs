using System.Collections.Generic;

public class MatchDecision
{
    public float MinimumActionConfidence { get; set; } = 0.20f;

    public float AmbiguousRelativeGap { get; set; } = 0.10f;

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

        // -----------------------------------------
        // INTENT
        // -----------------------------------------

        if (!top.IntendExplicit)
        {
            if (top.TargetExplicit)
                return MatchState.NeedIntend;

            return MatchState.NoMatch;
        }

        // -----------------------------------------
        // TARGET
        // -----------------------------------------

        if (!top.TargetExplicit)
        {
            return MatchState.NeedTarget;
        }

        // -----------------------------------------
        // Action confidence
        // -----------------------------------------

        if (top.FinalScore <
            MinimumActionConfidence)
        {
            return MatchState.NoMatch;
        }

        // -----------------------------------------
        // Ambiguous
        // -----------------------------------------

        if (results.Count >= 2)
        {
            MatchResult second =
                results[1];

            float gap =
                GetRelativeGap(
                    top.FinalScore,
                    second.FinalScore);

            if (gap <
                AmbiguousRelativeGap)
            {
                return MatchState.Ambiguous;
            }
        }

        return MatchState.Confident;
    }

    private float GetRelativeGap(
        float top,
        float second)
    {
        if (top <= 0f)
            return 0f;

        return
            (top - second) /
            top;
    }
}