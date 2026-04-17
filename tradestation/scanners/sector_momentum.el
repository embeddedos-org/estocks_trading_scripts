{ Sector Momentum Score — RadarScreen Indicator }
{ Composite momentum scoring with multi-timeframe ROC, relative strength, and volume trend }

Inputs:
    ROC1_Period(20),
    ROC1_Weight(0.40),
    ROC3_Period(63),
    ROC3_Weight(0.35),
    ROC6_Period(126),
    ROC6_Weight(0.25),
    VolFastLen(20),
    VolSlowLen(50),
    RSLookback(63),
    SPYSymbol("SPY");

Variables:
    ROC1(0),
    ROC3(0),
    ROC6(0),
    MomentumScore(0),
    VolFastAvg(0),
    VolSlowAvg(0),
    VolTrend(0),
    SymPerformance(0),
    SPYClose(0),
    SPYClosePrev(0),
    SPYPerformance(0),
    RelStrength(0),
    CompositeScore(0),
    CellColor(White);

{ 1-Month Rate of Change }
if Close[ROC1_Period] > 0 then
    ROC1 = (Close - Close[ROC1_Period]) / Close[ROC1_Period] * 100
else
    ROC1 = 0;

{ 3-Month Rate of Change }
if Close[ROC3_Period] > 0 then
    ROC3 = (Close - Close[ROC3_Period]) / Close[ROC3_Period] * 100
else
    ROC3 = 0;

{ 6-Month Rate of Change }
if Close[ROC6_Period] > 0 then
    ROC6 = (Close - Close[ROC6_Period]) / Close[ROC6_Period] * 100
else
    ROC6 = 0;

{ Weighted Momentum Score }
MomentumScore = ROC1 * ROC1_Weight + ROC3 * ROC3_Weight + ROC6 * ROC6_Weight;

{ Volume Trend: ratio of fast avg volume to slow avg volume }
VolFastAvg = Average(Volume, VolFastLen);
VolSlowAvg = Average(Volume, VolSlowLen);
if VolSlowAvg > 0 then
    VolTrend = (VolFastAvg / VolSlowAvg - 1) * 100
else
    VolTrend = 0;

{ Relative Strength vs SPY }
SPYClose = Close of Data2;
SPYClosePrev = Close[RSLookback] of Data2;

if Close[RSLookback] > 0 then
    SymPerformance = (Close - Close[RSLookback]) / Close[RSLookback] * 100
else
    SymPerformance = 0;

if SPYClosePrev > 0 then
    SPYPerformance = (SPYClose - SPYClosePrev) / SPYClosePrev * 100
else
    SPYPerformance = 0;

RelStrength = SymPerformance - SPYPerformance;

{ Composite Score }
CompositeScore = MomentumScore + RelStrength * 0.5 + VolTrend * 0.1;

{ Color Coding }
if CompositeScore > 10 then
    CellColor = DarkGreen
else if CompositeScore > 5 then
    CellColor = Green
else if CompositeScore >= -5 then
    CellColor = Yellow
else if CompositeScore >= -10 then
    CellColor = RGB(255, 165, 0) { Orange }
else
    CellColor = Red;

Plot1(CompositeScore, "MomScore");
SetPlotColor(1, CellColor);

Plot2(MomentumScore, "WtdROC");
Plot3(RelStrength, "RelStr");
Plot4(VolTrend, "VolTrnd");
