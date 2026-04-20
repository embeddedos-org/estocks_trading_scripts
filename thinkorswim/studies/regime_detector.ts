# Regime Detector Study
# ADX-based market regime classification with background coloring
# Scanner-compatible: outputs regime state as numeric value
# Regimes: TRENDING (ADX > 25), RANGING (ADX < 20), VOLATILE (ATR > 1.5x avg)

declare lower;

input ADXLength = 14;
input TrendThreshold = 25;
input RangeThreshold = 20;
input ATRLength = 14;
input VolatilityMultiplier = 1.5;
input ATRAvgLength = 50;

# ─── ADX Calculation ───
def HiDiff = high - high[1];
def LoDiff = low[1] - low;
def PlusDM = if HiDiff > LoDiff and HiDiff > 0 then HiDiff else 0;
def MinusDM = if LoDiff > HiDiff and LoDiff > 0 then LoDiff else 0;

def ATR = WildersAverage(TrueRange(high, close, low), ADXLength);
def PlusDI = 100 * WildersAverage(PlusDM, ADXLength) / ATR;
def MinusDI = 100 * WildersAverage(MinusDM, ADXLength) / ATR;
def DX = if (PlusDI + MinusDI > 0) then 100 * AbsValue(PlusDI - MinusDI) / (PlusDI + MinusDI) else 0;
def ADXValue = WildersAverage(DX, ADXLength);

# ─── Volatility Detection ───
def ATRCurrent = WildersAverage(TrueRange(high, close, low), ATRLength);
def ATRAvg = Average(ATRCurrent, ATRAvgLength);
def IsVolatile = ATRCurrent > ATRAvg * VolatilityMultiplier;

# ─── Regime Classification ───
# 2 = TRENDING, 1 = RANGING, 0 = TRANSITIONAL, -1 = VOLATILE
def RegimeValue =
    if IsVolatile then -1
    else if ADXValue > TrendThreshold then 2
    else if ADXValue < RangeThreshold then 1
    else 0;

# ─── Plots ───
plot ADX = ADXValue;
ADX.SetDefaultColor(Color.CYAN);
ADX.SetLineWeight(2);

plot TrendLine = TrendThreshold;
TrendLine.SetDefaultColor(Color.GREEN);
TrendLine.SetStyle(Curve.LONG_DASH);

plot RangeLine = RangeThreshold;
RangeLine.SetDefaultColor(Color.BLUE);
RangeLine.SetStyle(Curve.LONG_DASH);

# ─── Regime State Plot (Scanner-compatible) ───
plot Regime = RegimeValue;
Regime.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Regime.AssignValueColor(
    if RegimeValue == 2 then Color.GREEN
    else if RegimeValue == 1 then Color.BLUE
    else if RegimeValue == -1 then Color.RED
    else Color.GRAY
);
Regime.SetLineWeight(3);
Regime.Hide();

# ─── Regime Label ───
AddLabel(yes,
    if RegimeValue == 2 then "REGIME: TRENDING"
    else if RegimeValue == 1 then "REGIME: RANGING"
    else if RegimeValue == -1 then "REGIME: VOLATILE"
    else "REGIME: TRANSITIONAL",
    if RegimeValue == 2 then Color.GREEN
    else if RegimeValue == 1 then Color.BLUE
    else if RegimeValue == -1 then Color.RED
    else Color.GRAY
);

AddLabel(yes, "ADX: " + Round(ADXValue, 1), Color.CYAN);
AddLabel(yes, "ATR: " + Round(ATRCurrent, 2), Color.YELLOW);

# ─── Background Coloring ───
AssignBackgroundColor(
    if RegimeValue == 2 then CreateColor(0, 100, 0)
    else if RegimeValue == 1 then CreateColor(0, 0, 120)
    else if RegimeValue == -1 then CreateColor(120, 60, 0)
    else CreateColor(60, 60, 60)
);

# ─── Scanner Columns ───
# Use "Regime" plot in Stock Hacker:
#   Regime equals 2  → find trending stocks
#   Regime equals 1  → find ranging stocks
#   Regime equals -1 → find volatile stocks
