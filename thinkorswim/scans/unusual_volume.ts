# Unusual Volume Scanner
# Detects stocks with unusually high volume relative to their average,
# confirmed by significant price movement

# --- Inputs ---
input avgVolumePeriod = 20;
input volumeRatioThreshold = 2.0;
input priceMovementPct = 0.5;
input showAlerts = yes;

# --- Volume Analysis ---
def avgVolume = Average(volume[1], avgVolumePeriod);
def volRatio = if avgVolume > 0 then volume / avgVolume else 0;

# --- Price Movement Confirmation ---
def priceChange = if open != 0 then AbsValue(close - open) / open * 100 else 0;
def priceChangeRaw = if open != 0 then (close - open) / open * 100 else 0;

# --- Direction Detection ---
def isUpVolume = close > open;
def isDownVolume = close < open;
def direction = if isUpVolume then 1 else if isDownVolume then -1 else 0;

# --- Filter Conditions ---
def unusualVolumeCondition = volRatio >= volumeRatioThreshold;
def priceMovementCondition = priceChange >= priceMovementPct;
def allConditionsMet = unusualVolumeCondition and priceMovementCondition;

# --- Main Scan Result: Volume Ratio ---
plot VolumeRatio = if allConditionsMet then volRatio else Double.NaN;
VolumeRatio.SetDefaultColor(Color.WHITE);
VolumeRatio.SetLineWeight(2);

# --- Direction Indicator ---
plot DirectionSignal = if allConditionsMet then direction else Double.NaN;
DirectionSignal.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
DirectionSignal.SetLineWeight(3);
DirectionSignal.DefineColor("UpVolume", Color.GREEN);
DirectionSignal.DefineColor("DownVolume", Color.RED);
DirectionSignal.DefineColor("Neutral", Color.GRAY);
DirectionSignal.AssignValueColor(
    if direction > 0 then DirectionSignal.Color("UpVolume")
    else if direction < 0 then DirectionSignal.Color("DownVolume")
    else DirectionSignal.Color("Neutral")
);

# --- Volume Bar Visualization ---
plot VolBar = if allConditionsMet then volume else Double.NaN;
VolBar.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
VolBar.AssignValueColor(
    if isUpVolume then Color.GREEN
    else if isDownVolume then Color.RED
    else Color.GRAY
);

# --- Average Volume Reference ---
plot AvgVolLine = avgVolume;
AvgVolLine.SetDefaultColor(Color.YELLOW);
AvgVolLine.SetStyle(Curve.SHORT_DASH);
AvgVolLine.SetLineWeight(1);

# --- Background Color ---
AssignBackgroundColor(
    if allConditionsMet and isUpVolume then Color.DARK_GREEN
    else if allConditionsMet and isDownVolume then Color.DARK_RED
    else Color.CURRENT
);

# --- Threshold Line ---
plot ThresholdLine = avgVolume * volumeRatioThreshold;
ThresholdLine.SetDefaultColor(Color.ORANGE);
ThresholdLine.SetStyle(Curve.LONG_DASH);

# --- Scan-compatible boolean plot ---
plot ScanFilter = allConditionsMet;
ScanFilter.Hide();

# --- Breakout Intensity Score ---
# Combines volume ratio and price movement for ranking
def intensityScore = if allConditionsMet then volRatio * priceChange else 0;

plot Intensity = if allConditionsMet then intensityScore else Double.NaN;
Intensity.SetDefaultColor(Color.CYAN);
Intensity.Hide();

# --- Labels ---
AddLabel(yes, "Vol Ratio: " + Round(volRatio, 2) + "x",
    if volRatio >= volumeRatioThreshold then Color.GREEN else Color.GRAY);

AddLabel(yes, "Avg Vol(" + avgVolumePeriod + "): " + Round(avgVolume, 0), Color.YELLOW);

AddLabel(yes, "Price Move: " + Round(priceChangeRaw, 2) + "%",
    if priceChangeRaw > 0 then Color.GREEN
    else if priceChangeRaw < 0 then Color.RED
    else Color.GRAY);

AddLabel(allConditionsMet,
    if isUpVolume then "▲ UNUSUAL UP VOLUME"
    else if isDownVolume then "▼ UNUSUAL DOWN VOLUME"
    else "— UNUSUAL VOLUME",
    if isUpVolume then Color.GREEN else Color.RED);

# --- Alerts ---
Alert(showAlerts and allConditionsMet and isUpVolume,
    "Unusual UP Volume: " + Round(volRatio, 1) + "x avg, +" + Round(priceChangeRaw, 2) + "%",
    Alert.BAR, Sound.Ding);

Alert(showAlerts and allConditionsMet and isDownVolume,
    "Unusual DOWN Volume: " + Round(volRatio, 1) + "x avg, " + Round(priceChangeRaw, 2) + "%",
    Alert.BAR, Sound.Ring);
