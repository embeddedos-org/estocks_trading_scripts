{ Kaufman Adaptive Moving Average (KAMA) Indicator }

Inputs:
    Length(10),
    FastLength(2),
    SlowLength(30),
    Price(Close);

Variables:
    Direction(0),
    Volatility(0),
    ER(0),
    FastSC(0),
    SlowSC(0),
    SC(0),
    KAMA(0),
    PrevKAMA(0),
    i(0),
    KAMAColor(White);

FastSC = 2 / (FastLength + 1);
SlowSC = 2 / (SlowLength + 1);

{ Efficiency Ratio: absolute price change over N periods divided by sum of absolute period-to-period changes }
Direction = AbsValue(Price - Price[Length]);

Volatility = 0;
for i = 0 to Length - 1 begin
    Volatility = Volatility + AbsValue(Price[i] - Price[i + 1]);
end;

if Volatility > 0 then
    ER = Direction / Volatility
else
    ER = 0;

{ Smoothing Constant }
SC = Power(ER * (FastSC - SlowSC) + SlowSC, 2);

{ KAMA Calculation }
if CurrentBar = 1 then
    KAMA = Price
else begin
    PrevKAMA = KAMA[1];
    KAMA = PrevKAMA + SC * (Price - PrevKAMA);
end;

{ Color: green when KAMA rising, red when falling }
if KAMA > KAMA[1] then
    KAMAColor = Green
else if KAMA < KAMA[1] then
    KAMAColor = Red
else
    KAMAColor = Yellow;

Plot1(KAMA, "KAMA");
SetPlotColor(1, KAMAColor);
SetPlotWidth(1, 2);

Plot2(Price, "Close");
SetPlotColor(2, White);
SetPlotWidth(2, 1);
