# Earnings Play Strategy
# Pre-earnings momentum strategy: enters before earnings when IV rank is elevated,
# exits before the announcement to avoid binary event risk

declare lower;

# --- Inputs ---
input daysBeforeEarnings = 5;
input ivRankThreshold = 50;
input positionSize = 100;
input showLabels = yes;
input showEarningsMarkers = yes;
input ivLookbackPeriod = 252;

# --- Implied Volatility Data ---
def currentIV = if !IsNaN(imp_volatility()) then imp_volatility() * 100 else 0;

# --- IV Rank Calculation ---
# IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
def ivHigh = Highest(currentIV, ivLookbackPeriod);
def ivLow = Lowest(currentIV, ivLookbackPeriod);
def ivRange = ivHigh - ivLow;
def ivRank = if ivRange > 0 then (currentIV - ivLow) / ivRange * 100 else 0;

# --- IV Percentile (alternative measure) ---
# Count how many days IV was below current level
rec ivBelowCount = if currentIV[1] < currentIV then ivBelowCount[1] + 1 else ivBelowCount[1];
rec totalBars = totalBars[1] + 1;
def ivPercentile = if totalBars > 0 then ivBelowCount / totalBars * 100 else 0;

# --- Earnings Detection ---
def hasEarningsToday = HasEarnings();
def hasEarningsInPeriod = Sum(HasEarnings(), daysBeforeEarnings) > 0;

# --- Days Until Earnings ---
# Forward-scan to find next earnings date
def daysToEarnings = fold i = 1 to daysBeforeEarnings + 2
    with counter = 0
    while counter == 0
    do if GetValue(HasEarnings(), -i) then i else 0;

def earningsWithinRange = daysToEarnings > 0 and daysToEarnings <= daysBeforeEarnings;
def earningsTomorrow = daysToEarnings == 1;

# --- Entry / Exit Conditions ---
def entryCondition = earningsWithinRange
    and !earningsTomorrow
    and ivRank >= ivRankThreshold
    and close > Average(close, 20);

def exitCondition = earningsTomorrow or hasEarningsToday;

# --- Position Tracking ---
rec inPosition = if entryCondition and !inPosition[1] then 1
    else if exitCondition and inPosition[1] then 0
    else inPosition[1];

def entrySignal = entryCondition and !inPosition[1];
def exitSignal = exitCondition and inPosition[1];

# --- Entry Price Tracking ---
rec entryPrice = if entrySignal then close else if inPosition then entryPrice[1] else 0;

# --- Orders ---
AddOrder(OrderType.BUY_TO_OPEN,
    entrySignal,
    close,
    positionSize,
    Color.GREEN,
    Color.GREEN,
    "Earnings Entry");

AddOrder(OrderType.SELL_TO_CLOSE,
    exitSignal,
    close,
    positionSize,
    Color.RED,
    Color.RED,
    "Pre-Earnings Exit");

# --- P&L Tracking ---
def unrealizedPL = if inPosition then (close - entryPrice) * positionSize else 0;
rec realizedPL = if exitSignal then realizedPL[1] + (close - entryPrice) * positionSize
    else realizedPL[1];
rec tradeCount = if exitSignal then tradeCount[1] + 1 else tradeCount[1];
rec winCount = if exitSignal and close > entryPrice then winCount[1] + 1 else winCount[1];

# --- Plots ---

# IV Rank Plot
plot IVRank = ivRank;
IVRank.SetDefaultColor(Color.CYAN);
IVRank.SetLineWeight(2);
IVRank.AssignValueColor(
    if ivRank >= 80 then Color.RED
    else if ivRank >= ivRankThreshold then Color.YELLOW
    else Color.GREEN
);

# IV Rank Threshold Line
plot IVThreshold = ivRankThreshold;
IVThreshold.SetDefaultColor(Color.ORANGE);
IVThreshold.SetStyle(Curve.SHORT_DASH);

# Reference Lines
plot Level80 = 80;
Level80.SetDefaultColor(Color.DARK_RED);
Level80.SetStyle(Curve.SHORT_DASH);

plot Level20 = 20;
Level20.SetDefaultColor(Color.DARK_GREEN);
Level20.SetStyle(Curve.SHORT_DASH);

# Entry Markers
plot EntryPoint = if entrySignal then ivRank else Double.NaN;
EntryPoint.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
EntryPoint.SetDefaultColor(Color.GREEN);
EntryPoint.SetLineWeight(4);

# Exit Markers
plot ExitPoint = if exitSignal then ivRank else Double.NaN;
ExitPoint.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
ExitPoint.SetDefaultColor(Color.RED);
ExitPoint.SetLineWeight(4);

# Earnings Day Markers
plot EarningsMarker = if showEarningsMarkers and hasEarningsToday then ivRank else Double.NaN;
EarningsMarker.SetPaintingStrategy(PaintingStrategy.POINTS);
EarningsMarker.SetDefaultColor(Color.MAGENTA);
EarningsMarker.SetLineWeight(5);

# In-Position Shading
AddCloud(if inPosition then 100 else Double.NaN, 0, CreateColor(0, 50, 0), CreateColor(0, 50, 0));

# Earnings Bubble
AddChartBubble(showEarningsMarkers and hasEarningsToday, ivRank, "EARNINGS", Color.MAGENTA, yes);

# Entry/Exit Bubbles
AddChartBubble(entrySignal, ivRank,
    "BUY\nIVR:" + Round(ivRank, 1) + "\n$" + Round(close, 2),
    Color.GREEN, no);

AddChartBubble(exitSignal, ivRank,
    "SELL\nP&L:$" + Round((close - entryPrice) * positionSize, 2),
    if close >= entryPrice then Color.GREEN else Color.RED, yes);

# --- Labels ---
AddLabel(showLabels, "IV Rank: " + Round(ivRank, 1) + "%",
    if ivRank >= 80 then Color.RED
    else if ivRank >= ivRankThreshold then Color.YELLOW
    else Color.GREEN);

AddLabel(showLabels, "Current IV: " + Round(currentIV, 1) + "%", Color.CYAN);

AddLabel(showLabels, "IV Range: " + Round(ivLow, 1) + " - " + Round(ivHigh, 1), Color.GRAY);

AddLabel(showLabels and inPosition, "IN POSITION @ $" + Round(entryPrice, 2), Color.GREEN);
AddLabel(showLabels and inPosition, "Unrealized P&L: $" + Round(unrealizedPL, 2),
    if unrealizedPL >= 0 then Color.GREEN else Color.RED);

AddLabel(showLabels, "Realized P&L: $" + Round(realizedPL, 2),
    if realizedPL >= 0 then Color.GREEN else Color.RED);

AddLabel(showLabels and tradeCount > 0,
    "Trades: " + tradeCount + " | Win Rate: " + Round(if tradeCount > 0 then winCount / tradeCount * 100 else 0, 1) + "%",
    Color.WHITE);

AddLabel(showLabels and daysToEarnings > 0,
    "Earnings in " + daysToEarnings + " days",
    Color.MAGENTA);

# --- Alerts ---
Alert(entrySignal, "Earnings Play Entry: IV Rank " + Round(ivRank, 1) + "%, " + daysToEarnings + " days to earnings", Alert.BAR, Sound.Ding);
Alert(exitSignal, "Earnings Play Exit: Close position before earnings", Alert.BAR, Sound.Ring);
Alert(hasEarningsToday, "EARNINGS TODAY", Alert.BAR, Sound.Bell);
