{ Regime Switcher Strategy }
{ ADX-based regime detection switching between trend-following and mean-reversion }
{ Trend: MA crossover with Chandelier Exit }
{ Range: RSI extremes with Bollinger Band exits }

Inputs:
    { Regime Detection }
    ADXLength(14),
    TrendThreshold(25),
    RangeThreshold(20),
    ATRPeriod(14),
    VolatilityMult(1.5),

    { Trend Strategy }
    FastMALength(9),
    SlowMALength(21),
    TrendATRMult(3.0),

    { Mean Reversion }
    BBLength(20),
    BBDeviation(2.0),
    RSILength(14),
    RSIOverbought(70),
    RSIOversold(30),

    { Risk }
    RiskPercent(2);

Variables:
    FastMA(0),
    SlowMA(0),
    ADXValue(0),
    ATRValue(0),
    ATRAverage(0),
    IsVolatile(false),
    IsTrending(false),
    IsRanging(false),
    RegimeStr(""),

    { Bollinger Bands }
    BBMid(0),
    BBUpper(0),
    BBLower(0),

    { RSI }
    RSIValue(0),

    { Chandelier Exit }
    HighestHigh(0),
    LowestLow(0),
    ChandelierLong(0),
    ChandelierShort(0),

    { Position Sizing }
    RiskAmount(0),
    StopDistance(0),
    ShareSize(0);

{ ─── Regime Detection ─── }
ADXValue = ADX(ADXLength);
ATRValue = AvgTrueRange(ATRPeriod);
ATRAverage = Average(ATRValue, 50);
IsVolatile = ATRValue > ATRAverage * VolatilityMult;
IsTrending = ADXValue > TrendThreshold and IsVolatile = false;
IsRanging = ADXValue < RangeThreshold and IsVolatile = false;

if IsTrending then
    RegimeStr = "TRENDING"
else if IsRanging then
    RegimeStr = "RANGING"
else if IsVolatile then
    RegimeStr = "VOLATILE"
else
    RegimeStr = "TRANSITION";

{ ─── Trend Strategy Components ─── }
FastMA = Average(Close, FastMALength);
SlowMA = Average(Close, SlowMALength);

HighestHigh = Highest(High, ATRPeriod);
LowestLow = Lowest(Low, ATRPeriod);
ChandelierLong = HighestHigh - TrendATRMult * ATRValue;
ChandelierShort = LowestLow + TrendATRMult * ATRValue;

{ ─── Mean Reversion Components ─── }
BBMid = Average(Close, BBLength);
BBUpper = BBMid + BBDeviation * StdDev(Close, BBLength);
BBLower = BBMid - BBDeviation * StdDev(Close, BBLength);
RSIValue = RSI(Close, RSILength);

{ ─── Position Sizing ─── }
if ATRValue > 0 then begin
    RiskAmount = (RiskPercent / 100) * Portfolio_Equity;
    StopDistance = TrendATRMult * ATRValue;
    ShareSize = IntPortion(RiskAmount / StopDistance);
    if ShareSize < 1 then
        ShareSize = 1;
end;

{ ─── Entry Logic ─── }
if MarketPosition = 0 then begin

    { Trend-Following: MA crossover when trending }
    if IsTrending then begin
        if FastMA crosses above SlowMA then
            Buy("Trend Long") ShareSize shares next bar at market;

        if FastMA crosses below SlowMA then
            SellShort("Trend Short") ShareSize shares next bar at market;
    end;

    { Mean Reversion: RSI extremes at BB boundaries when ranging }
    if IsRanging then begin
        if Close <= BBLower and RSIValue < RSIOversold then
            Buy("MR Long") ShareSize shares next bar at market;

        if Close >= BBUpper and RSIValue > RSIOverbought then
            SellShort("MR Short") ShareSize shares next bar at market;
    end;
end;

{ ─── Exit Logic ─── }
{ Trend exits: Chandelier trailing stop }
if MarketPosition = 1 and IsTrending then begin
    Sell("Chand LX") next bar at ChandelierLong stop;
end;

if MarketPosition = -1 and IsTrending then begin
    BuyToCover("Chand SX") next bar at ChandelierShort stop;
end;

{ Mean Reversion exits: return to BB midline }
if MarketPosition = 1 and IsRanging then begin
    if Close >= BBMid then
        Sell("MR Exit Long") next bar at market;
end;

if MarketPosition = -1 and IsRanging then begin
    if Close <= BBMid then
        BuyToCover("MR Exit Short") next bar at market;
end;

{ Regime change exit: close if regime shifts to volatile }
if IsVolatile and MarketPosition <> 0 then begin
    if MarketPosition = 1 then
        Sell("Vol Exit L") next bar at market;
    if MarketPosition = -1 then
        BuyToCover("Vol Exit S") next bar at market;
end;

{ ─── Dashboard Output ─── }
Print("Regime: ", RegimeStr, " | ADX: ", NumToStr(ADXValue, 1),
      " | RSI: ", NumToStr(RSIValue, 1), " | ATR: ", NumToStr(ATRValue, 2),
      " | Pos: ", NumToStr(MarketPosition, 0));
