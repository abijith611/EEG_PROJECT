>>> DEBUG: DECODING STATISTICS <<<

=== Loaded 62 subjects ===
Subjects with any NaN: 0 / 62

==================================================
   OVERALL ACCURACY (Average across all 5 seconds)
   Chance level = 33.33%
==================================================

Own response:
  Overall: 34.64%
  Winners: 34.58%
  Losers:  34.70%
  -> Diff: -0.12%

Opponent's response:
  Overall: 33.38%
  Winners: 33.79%
  Losers:  32.96%
  -> Diff: +0.83%

Own previous response:
  Overall: 34.04%
  Winners: 34.00%
  Losers:  34.08%
  -> Diff: -0.08%

Opponent's previous response:
  Overall: 34.38%
  Winners: 34.04%
  Losers:  34.71%
  -> Diff: -0.67%

==================================================
Debug plot saved to: project/ds006761\derivatives\debug_decoding.png

>>> BAYES FACTOR OUTPUT FOR REPORT <<<

===========================================================================
 🏆 WINNERS VS. LOSERS STATS (USING R BAYESFACTOR) 🏆
===========================================================================

--- Own response ---
  Decision (0-2s):
    Winners -> Peak Acc: 35.06% | Max BF10: 0.33
    Losers  -> Peak Acc: 34.22% | Max BF10: 0.02
  Response (2-4s):
    Winners -> Peak Acc: 37.00% | Max BF10: 21.30
    Losers  -> Peak Acc: 37.59% | Max BF10: 63.46
  Feedback (4-5s):
    Winners -> Peak Acc: 35.83% | Max BF10: 10.87
    Losers  -> Peak Acc: 36.20% | Max BF10: 62.14

--- Opponent's response ---
  Decision (0-2s):
    Winners -> Peak Acc: 34.08% | Max BF10: 0.01
    Losers  -> Peak Acc: 33.69% | Max BF10: 0.00
  Response (2-4s):
    Winners -> Peak Acc: 34.53% | Max BF10: 0.13
    Losers  -> Peak Acc: 33.69% | Max BF10: 0.00
  Feedback (4-5s):
    Winners -> Peak Acc: 37.11% | Max BF10: 152.04
    Losers  -> Peak Acc: 35.75% | Max BF10: 4.70

--- Own previous response ---
  Decision (0-2s):
    Winners -> Peak Acc: 34.69% | Max BF10: 0.08
    Losers  -> Peak Acc: 34.54% | Max BF10: 0.13
  Response (2-4s):
    Winners -> Peak Acc: 35.18% | Max BF10: 1.71
    Losers  -> Peak Acc: 35.94% | Max BF10: 35.84
  Feedback (4-5s):
    Winners -> Peak Acc: 34.11% | Max BF10: 0.02
    Losers  -> Peak Acc: 34.58% | Max BF10: 0.28

--- Opponent's previous response ---
  Decision (0-2s):
    Winners -> Peak Acc: 36.27% | Max BF10: 23.23
    Losers  -> Peak Acc: 36.42% | Max BF10: 114.76
  Response (2-4s):
    Winners -> Peak Acc: 34.76% | Max BF10: 0.11
    Losers  -> Peak Acc: 34.47% | Max BF10: 0.08
  Feedback (4-5s):
    Winners -> Peak Acc: 34.50% | Max BF10: 0.06
    Losers  -> Peak Acc: 35.78% | Max BF10: 8.36

==================================================
Pipeline Complete! Check 'results/plots' for results.
==================================================