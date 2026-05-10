# VWAP with Standard Deviation Bands
# Session-anchored VWAP with ±1σ, ±2σ, ±3σ deviation bands
# Lower study: Distance-from-VWAP
# Alerts when price reaches ±2σ

declare upper;

input ShowBand1 = yes;
input ShowBand2 = yes;
input ShowBand3 = yes;
input PriceSource = FundamentalType.HLC3;

# ─── VWAP Calculation ───
def isNewDay = GetDay() != GetDay()[1];
def Price = FundamentalValue(PriceSource);

def cumVol = if isNewDay then volume else cumVol[1] + volume;
def cumTP = if isNewDay then Price * volume else cumTP[1] + Price * volume;
def cumTP2 = if isNewDay then Price * Price * volume else cumTP2[1] + Price * Price * volume;

def VWAPValue = if cumVol > 0 then cumTP / cumVol else close;
def Variance = if cumVol > 0 then Max(cumTP2 / cumVol - VWAPValue * VWAPValue, 0) else 0;
def StDev = Sqrt(Variance);

# ─── VWAP Plot ───
plot VWAP = VWAPValue;
VWAP.SetDefaultColor(Color.YELLOW);
VWAP.SetLineWeight(2);

# ─── Band 1: ±1σ ───
plot Band1Up = if ShowBand1 then VWAPValue + StDev else Double.NaN;
plot Band1Dn = if ShowBand1 then VWAPValue - StDev else Double.NaN;
Band1Up.SetDefaultColor(Color.CYAN);
Band1Dn.SetDefaultColor(Color.CYAN);
Band1Up.SetStyle(Curve.SHORT_DASH);
Band1Dn.SetStyle(Curve.SHORT_DASH);

# ─── Band 2: ±2σ ───
plot Band2Up = if ShowBand2 then VWAPValue + StDev * 2 else Double.NaN;
plot Band2Dn = if ShowBand2 then VWAPValue - StDev * 2 else Double.NaN;
Band2Up.SetDefaultColor(Color.ORANGE);
Band2Dn.SetDefaultColor(Color.ORANGE);
Band2Up.SetStyle(Curve.SHORT_DASH);
Band2Dn.SetStyle(Curve.SHORT_DASH);

# ─── Band 3: ±3σ ───
plot Band3Up = if ShowBand3 then VWAPValue + StDev * 3 else Double.NaN;
plot Band3Dn = if ShowBand3 then VWAPValue - StDev * 3 else Double.NaN;
Band3Up.SetDefaultColor(Color.RED);
Band3Dn.SetDefaultColor(Color.RED);
Band3Up.SetStyle(Curve.SHORT_DASH);
Band3Dn.SetStyle(Curve.SHORT_DASH);

# ─── Cloud Fills ───
AddCloud(Band1Up, VWAP, Color.DARK_GREEN, Color.DARK_GREEN);
AddCloud(VWAP, Band1Dn, Color.DARK_GREEN, Color.DARK_GREEN);
AddCloud(Band2Up, Band1Up, Color.DARK_ORANGE, Color.DARK_ORANGE);
AddCloud(Band1Dn, Band2Dn, Color.DARK_ORANGE, Color.DARK_ORANGE);

# ─── Distance from VWAP Labels ───
def DistPct = if VWAPValue > 0 then (close - VWAPValue) / VWAPValue * 100 else 0;
def DistSD = if StDev > 0 then (close - VWAPValue) / StDev else 0;

AddLabel(yes, "VWAP: " + Round(VWAPValue, 2), Color.YELLOW);
AddLabel(yes, "Dist: " + Round(DistPct, 2) + "%",
    if DistPct > 0 then Color.GREEN else Color.RED);
AddLabel(yes, "σ: " + Round(DistSD, 2),
    if AbsValue(DistSD) > 2 then Color.RED
    else if AbsValue(DistSD) > 1 then Color.ORANGE
    else Color.GREEN);

# ─── Alerts ───
Alert(close crosses above Band2Up, "Price above VWAP +2σ", Alert.BAR, Sound.Ding);
Alert(close crosses below Band2Dn, "Price below VWAP -2σ", Alert.BAR, Sound.Ding);
Alert(close crosses above Band3Up, "Price above VWAP +3σ", Alert.BAR, Sound.Ring);
Alert(close crosses below Band3Dn, "Price below VWAP -3σ", Alert.BAR, Sound.Ring);
