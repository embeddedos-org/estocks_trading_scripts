# Mansfield Relative Strength Study
# Measures a stock's performance relative to a benchmark index
# Based on Stan Weinstein's Mansfield Relative Strength methodology

declare lower;

# --- Inputs ---
input referenceSymbol = "SPX";
input maLength = 52;
input averageType = AverageType.SIMPLE;
input showBackgroundColoring = yes;
input showLabels = yes;
input dailyBarsPerWeek = 5;

# --- Raw Relative Strength Calculation ---
def refClose = close(referenceSymbol);
def rawRS = if refClose != 0 then close / refClose else 0;

# --- Mansfield Relative Strength ---
# Mansfield RS = ((RS / SMA(RS, period)) - 1) * 100
def maLengthBars = maLength * dailyBarsPerWeek;
def rsMA = SimpleMovingAvg(rawRS, maLengthBars);
def mansfieldRS = if rsMA != 0 then ((rawRS / rsMA) - 1) * 100 else 0;

# --- RS Moving Average ---
def rsSmoothLength = Round(maLengthBars / 4, 0);
def rsSmoothed = SimpleMovingAvg(mansfieldRS, rsSmoothLength);

# --- Plots ---
plot RSLine = mansfieldRS;
RSLine.SetDefaultColor(Color.CYAN);
RSLine.SetLineWeight(2);

plot RSMovingAvg = rsSmoothed;
RSMovingAvg.SetDefaultColor(Color.YELLOW);
RSMovingAvg.SetLineWeight(1);
RSMovingAvg.SetStyle(Curve.SHORT_DASH);

plot ZeroLine = 0;
ZeroLine.SetDefaultColor(Color.GRAY);
ZeroLine.SetStyle(Curve.SHORT_DASH);

# --- RS Line Coloring ---
RSLine.AssignValueColor(
    if mansfieldRS > rsSmoothed and mansfieldRS > 0 then Color.GREEN
    else if mansfieldRS > rsSmoothed and mansfieldRS <= 0 then Color.DARK_GREEN
    else if mansfieldRS <= rsSmoothed and mansfieldRS > 0 then Color.DARK_RED
    else Color.RED
);

# --- Background Coloring ---
AddCloud(if showBackgroundColoring and mansfieldRS > rsSmoothed then mansfieldRS else Double.NaN, rsSmoothed, Color.DARK_GREEN, Color.DARK_GREEN);
AddCloud(if showBackgroundColoring and mansfieldRS <= rsSmoothed then rsSmoothed else Double.NaN, mansfieldRS, Color.DARK_RED, Color.DARK_RED);

# --- Trend Detection ---
def rsRising = mansfieldRS > mansfieldRS[1];
def rsFalling = mansfieldRS < mansfieldRS[1];
def rsAboveMA = mansfieldRS > rsSmoothed;
def rsBelowMA = mansfieldRS < rsSmoothed;
def rsAboveZero = mansfieldRS > 0;

# --- Crossover Signals ---
def rsCrossAboveMA = mansfieldRS crosses above rsSmoothed;
def rsCrossBelowMA = mansfieldRS crosses below rsSmoothed;
def rsCrossAboveZero = mansfieldRS crosses above 0;
def rsCrossBelowZero = mansfieldRS crosses below 0;

plot RSCrossUpSignal = if rsCrossAboveMA then mansfieldRS else Double.NaN;
RSCrossUpSignal.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
RSCrossUpSignal.SetDefaultColor(Color.GREEN);
RSCrossUpSignal.SetLineWeight(2);

plot RSCrossDownSignal = if rsCrossBelowMA then mansfieldRS else Double.NaN;
RSCrossDownSignal.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
RSCrossDownSignal.SetDefaultColor(Color.RED);
RSCrossDownSignal.SetLineWeight(2);

# --- Scanner-Compatible Signal ---
# Signal > 0 = outperforming (RS above MA), Signal < 0 = underperforming
plot ScanSignal = if rsAboveMA and rsAboveZero then 2
    else if rsAboveMA and !rsAboveZero then 1
    else if rsBelowMA and rsAboveZero then -1
    else -2;
ScanSignal.Hide();

# --- Labels ---
AddLabel(showLabels, "RS vs " + referenceSymbol + ": " + Round(mansfieldRS, 2),
    if mansfieldRS > rsSmoothed then Color.GREEN else Color.RED);

AddLabel(showLabels,
    if rsAboveMA and rsAboveZero then "Outperforming ▲"
    else if rsAboveMA and !rsAboveZero then "Improving ↑"
    else if rsBelowMA and rsAboveZero then "Weakening ↓"
    else "Underperforming ▼",
    if rsAboveMA then Color.GREEN else Color.RED);

AddLabel(showLabels, "RS MA(" + maLength + "w): " + Round(rsSmoothed, 2), Color.YELLOW);

# --- Alerts ---
Alert(rsCrossAboveMA, "RS crossed above MA - Relative strength improving", Alert.BAR, Sound.Ding);
Alert(rsCrossBelowMA, "RS crossed below MA - Relative strength weakening", Alert.BAR, Sound.Ring);
Alert(rsCrossAboveZero, "RS crossed above zero - Now outperforming " + referenceSymbol, Alert.BAR, Sound.Ding);
Alert(rsCrossBelowZero, "RS crossed below zero - Now underperforming " + referenceSymbol, Alert.BAR, Sound.Ring);
