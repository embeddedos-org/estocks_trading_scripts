# Sector Rotation — Momentum Scoring Watchlist Column
# Calculates a weighted composite momentum score using Rate of Change (ROC)
# across multiple timeframes. Designed for sector ETFs.

# --- Inputs ---
input weeklyROCLength = 5;
input monthlyROCLength = 21;
input quarterlyROCLength = 63;
input weeklyWeight = 0.3;
input monthlyWeight = 0.4;
input quarterlyWeight = 0.3;
input smoothingLength = 3;

# --- Rate of Change Calculations ---
# 1-Week ROC (5 trading days)
def roc1W = if close[weeklyROCLength] != 0
    then (close - close[weeklyROCLength]) / close[weeklyROCLength] * 100
    else 0;

# 1-Month ROC (21 trading days)
def roc1M = if close[monthlyROCLength] != 0
    then (close - close[monthlyROCLength]) / close[monthlyROCLength] * 100
    else 0;

# 3-Month ROC (63 trading days)
def roc3M = if close[quarterlyROCLength] != 0
    then (close - close[quarterlyROCLength]) / close[quarterlyROCLength] * 100
    else 0;

# --- Weighted Composite Score ---
def rawScore = (roc1W * weeklyWeight) + (roc1M * monthlyWeight) + (roc3M * quarterlyWeight);

# --- Smoothed Score (reduce noise) ---
def compositeScore = Average(rawScore, smoothingLength);

# --- Main Plot: Sector Momentum Score ---
plot MomentumScore = Round(compositeScore, 2);

# --- Color-Coded Output ---
# Dark green: score > 5 (strong bullish momentum)
# Light green: score > 2 (moderate bullish momentum)
# Yellow: score between -2 and 2 (neutral)
# Orange: score < -2 (moderate bearish momentum)
# Red: score < -5 (strong bearish momentum)

MomentumScore.AssignValueColor(
    if compositeScore > 5 then CreateColor(0, 128, 0)
    else if compositeScore > 2 then CreateColor(144, 238, 144)
    else if compositeScore >= -2 then Color.YELLOW
    else if compositeScore >= -5 then Color.ORANGE
    else Color.RED
);

# --- Trend Direction ---
def scoreTrend = compositeScore - compositeScore[1];
def isAccelerating = scoreTrend > 0;
def isDecelerating = scoreTrend < 0;

# --- Momentum Regime Classification ---
def regime = if compositeScore > 5 then 3
    else if compositeScore > 2 then 2
    else if compositeScore > 0 then 1
    else if compositeScore > -2 then -1
    else if compositeScore > -5 then -2
    else -3;

# --- Individual ROC Component Plots (hidden, for data export) ---
plot ROC_1W = Round(roc1W, 2);
ROC_1W.Hide();

plot ROC_1M = Round(roc1M, 2);
ROC_1M.Hide();

plot ROC_3M = Round(roc3M, 2);
ROC_3M.Hide();

# --- Trend Arrow Indicator ---
plot TrendIndicator = if isAccelerating then 1 else -1;
TrendIndicator.Hide();

# --- Regime Plot (for sorting/filtering) ---
plot RegimePlot = regime;
RegimePlot.Hide();

# --- Labels ---
AddLabel(yes, "Momentum: " + Round(compositeScore, 2),
    if compositeScore > 5 then CreateColor(0, 128, 0)
    else if compositeScore > 2 then CreateColor(144, 238, 144)
    else if compositeScore >= -2 then Color.YELLOW
    else if compositeScore >= -5 then Color.ORANGE
    else Color.RED
);

AddLabel(yes, "1W: " + Round(roc1W, 2) + "%",
    if roc1W > 0 then Color.GREEN else Color.RED);

AddLabel(yes, "1M: " + Round(roc1M, 2) + "%",
    if roc1M > 0 then Color.GREEN else Color.RED);

AddLabel(yes, "3M: " + Round(roc3M, 2) + "%",
    if roc3M > 0 then Color.GREEN else Color.RED);

AddLabel(yes,
    if isAccelerating then "Accelerating ▲" else "Decelerating ▼",
    if isAccelerating then Color.GREEN else Color.RED);

# --- Rank Classification Label ---
AddLabel(yes,
    if compositeScore > 5 then "STRONG BUY"
    else if compositeScore > 2 then "BUY"
    else if compositeScore >= -2 then "NEUTRAL"
    else if compositeScore >= -5 then "SELL"
    else "STRONG SELL",
    if compositeScore > 5 then CreateColor(0, 128, 0)
    else if compositeScore > 2 then CreateColor(144, 238, 144)
    else if compositeScore >= -2 then Color.YELLOW
    else if compositeScore >= -5 then Color.ORANGE
    else Color.RED
);
