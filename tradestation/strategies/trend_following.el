{ Trend Following Strategy }
{ Dual Moving Average Crossover with ADX Filter and Chandelier Exit }

Inputs:
    FastMALength(20),
    SlowMALength(50),
    ADXLength(14),
    ADXThreshold(25),
    ATRPeriod(22),
    ATRMultiplier(3.0),
    RiskPercent(2);

Variables:
    FastMA(0),
    SlowMA(0),
    ADXValue(0),
    ATRValue(0),
    ChandelierLongStop(0),
    ChandelierShortStop(0),
    HighestHigh(0),
    LowestLow(0),
    RiskAmount(0),
    ShareSize(0),
    StopDistance(0),
    intrabarpersalivet(false);

FastMA = Average(Close, FastMALength);
SlowMA = Average(Close, SlowMALength);
ADXValue = ADX(ADXLength);
ATRValue = AvgTrueRange(ATRPeriod);

HighestHigh = Highest(High, ATRPeriod);
LowestLow = Lowest(Low, ATRPeriod);

ChandelierLongStop = HighestHigh - ATRMultiplier * ATRValue;
ChandelierShortStop = LowestLow + ATRMultiplier * ATRValue;

{ Fixed Fractional Position Sizing: risk RiskPercent% of equity per trade }
if ATRValue > 0 then begin
    RiskAmount = (RiskPercent / 100) * Portfolio_Equity;
    StopDistance = ATRMultiplier * ATRValue;
    ShareSize = IntPortion(RiskAmount / StopDistance);
    if ShareSize < 1 then
        ShareSize = 1;
end;

{ Entry Logic }
if MarketPosition = 0 then begin
    { Long Entry: fast MA crosses above slow MA with ADX confirmation }
    if FastMA crosses above SlowMA and ADXValue > ADXThreshold then
        Buy("TF Long") ShareSize shares next bar at market;

    { Short Entry: fast MA crosses below slow MA with ADX confirmation }
    if FastMA crosses below SlowMA and ADXValue > ADXThreshold then
        SellShort("TF Short") ShareSize shares next bar at market;
end;

{ Exit Logic }
if MarketPosition = 1 then begin
    { Exit long via Chandelier Exit trailing stop }
    Sell("Chand LX") next bar at ChandelierLongStop stop;
end;

if MarketPosition = -1 then begin
    { Exit short via Chandelier Exit trailing stop }
    BuyToCover("Chand SX") next bar at ChandelierShortStop stop;
end;
