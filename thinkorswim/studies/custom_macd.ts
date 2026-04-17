# Custom Enhanced MACD Study
# Enhanced MACD with histogram momentum coloring, crossover alerts,
# zero-line detection, and divergence detection

declare lower;

# --- Inputs ---
input fastLength = 12;
input slowLength = 26;
input signalLength = 9;
input averageType = AverageType.EXPONENTIAL;
input divergenceLookback = 50;
input showAlerts = yes;
input showDivergence = yes;

# --- MACD Calculation ---
def fastMA = MovAvgExponential(close, fastLength);
def slowMA = MovAvgExponential(close, slowLength);

plot MACDLine = fastMA - slowMA;
plot SignalLine = MovAvgExponential(MACDLine, signalLength);
plot Histogram = MACDLine - SignalLine;

MACDLine.SetDefaultColor(Color.CYAN);
MACDLine.SetLineWeight(2);
SignalLine.SetDefaultColor(Color.YELLOW);
SignalLine.SetLineWeight(1);

# --- Zero Line ---
plot ZeroLine = 0;
ZeroLine.SetDefaultColor(Color.GRAY);
ZeroLine.SetStyle(Curve.SHORT_DASH);

# --- Histogram Coloring (4 momentum states) ---
Histogram.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Histogram.DefineColor("PositiveIncreasing", Color.DARK_GREEN);
Histogram.DefineColor("PositiveDecreasing", Color.GREEN);
Histogram.DefineColor("NegativeDecreasing", Color.DARK_RED);
Histogram.DefineColor("NegativeIncreasing", Color.RED);

Histogram.AssignValueColor(
    if Histogram > 0 and Histogram > Histogram[1] then Histogram.Color("PositiveIncreasing")
    else if Histogram > 0 and Histogram <= Histogram[1] then Histogram.Color("PositiveDecreasing")
    else if Histogram < 0 and Histogram < Histogram[1] then Histogram.Color("NegativeDecreasing")
    else Histogram.Color("NegativeIncreasing")
);

# --- Signal Line Crossover Detection ---
def bullishCross = MACDLine crosses above SignalLine;
def bearishCross = MACDLine crosses below SignalLine;

plot BullCrossArrow = if bullishCross then MACDLine else Double.NaN;
BullCrossArrow.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
BullCrossArrow.SetDefaultColor(Color.GREEN);
BullCrossArrow.SetLineWeight(3);

plot BearCrossArrow = if bearishCross then MACDLine else Double.NaN;
BearCrossArrow.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
BearCrossArrow.SetDefaultColor(Color.RED);
BearCrossArrow.SetLineWeight(3);

# --- Zero-Line Cross Detection ---
def zeroCrossUp = MACDLine crosses above 0;
def zeroCrossDown = MACDLine crosses below 0;

AddChartBubble(zeroCrossUp, 0, "Zero+", Color.GREEN, yes);
AddChartBubble(zeroCrossDown, 0, "Zero-", Color.RED, no);

# --- Divergence Detection ---
# Find swing lows in price and MACD for bullish divergence
def priceAtLow = low;
def macdAtLow = MACDLine;

# Identify pivot lows (simple swing detection)
def isPricePivotLow = low < low[1] and low < low[2] and low < Lowest(low, 3)[1];
def isMACDPivotLow = MACDLine < MACDLine[1] and MACDLine < MACDLine[2];

# Track previous pivot low values using rec
rec prevPriceLow = if isPricePivotLow then priceAtLow else prevPriceLow[1];
rec prevMACDLow = if isMACDPivotLow then macdAtLow else prevMACDLow[1];
rec prevPriceLowBar = if isPricePivotLow then BarNumber() else prevPriceLowBar[1];

# Bullish divergence: price makes lower low, MACD makes higher low
def bullishDivergence = showDivergence
    and isPricePivotLow
    and priceAtLow < prevPriceLow
    and macdAtLow > prevMACDLow
    and BarNumber() - prevPriceLowBar[1] <= divergenceLookback
    and prevPriceLowBar[1] > 0;

# Find swing highs for bearish divergence
def isPricePivotHigh = high > high[1] and high > high[2] and high > Highest(high, 3)[1];
def isMACDPivotHigh = MACDLine > MACDLine[1] and MACDLine > MACDLine[2];

rec prevPriceHigh = if isPricePivotHigh then high else prevPriceHigh[1];
rec prevMACDHigh = if isMACDPivotHigh then MACDLine else prevMACDHigh[1];
rec prevPriceHighBar = if isPricePivotHigh then BarNumber() else prevPriceHighBar[1];

# Bearish divergence: price makes higher high, MACD makes lower high
def bearishDivergence = showDivergence
    and isPricePivotHigh
    and high > prevPriceHigh
    and MACDLine < prevMACDHigh
    and BarNumber() - prevPriceHighBar[1] <= divergenceLookback
    and prevPriceHighBar[1] > 0;

# Divergence markers
AddChartBubble(bullishDivergence, MACDLine, "Bull Div", Color.LIME, no);
AddChartBubble(bearishDivergence, MACDLine, "Bear Div", Color.MAGENTA, yes);

# --- Alerts ---
Alert(showAlerts and bullishCross, "MACD Bullish Crossover", Alert.BAR, Sound.Ding);
Alert(showAlerts and bearishCross, "MACD Bearish Crossover", Alert.BAR, Sound.Ring);
Alert(showAlerts and zeroCrossUp, "MACD Crossed Above Zero", Alert.BAR, Sound.Ding);
Alert(showAlerts and zeroCrossDown, "MACD Crossed Below Zero", Alert.BAR, Sound.Ring);
Alert(showAlerts and bullishDivergence, "Bullish MACD Divergence Detected", Alert.BAR, Sound.Bell);
Alert(showAlerts and bearishDivergence, "Bearish MACD Divergence Detected", Alert.BAR, Sound.Bell);

# --- Labels ---
AddLabel(yes, "MACD: " + Round(MACDLine, 4), if MACDLine > 0 then Color.GREEN else Color.RED);
AddLabel(yes, "Signal: " + Round(SignalLine, 4), Color.YELLOW);
AddLabel(yes, "Hist: " + Round(Histogram, 4),
    if Histogram > 0 and Histogram > Histogram[1] then Color.DARK_GREEN
    else if Histogram > 0 then Color.GREEN
    else if Histogram < 0 and Histogram < Histogram[1] then Color.DARK_RED
    else Color.RED
);
