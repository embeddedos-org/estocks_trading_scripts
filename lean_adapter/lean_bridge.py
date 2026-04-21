"""
QuantConnect LEAN Full Bridge
================================

Generates complete LEAN C# projects from Python strategy definitions.

Usage:
    from lean_adapter.lean_bridge import LEANProjectGenerator
    gen = LEANProjectGenerator()
    gen.generate_project(
        name="MyStrategy",
        symbols=["AAPL", "MSFT"],
        output_dir="./lean_projects/MyStrategy",
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlphaModelGenerator:
    """Generates LEAN AlphaModel C# code."""

    @staticmethod
    def generate(strategy_name: str, indicators: Optional[List[str]] = None) -> str:
        indicators = indicators or ["RSI", "EMA"]
        ind_decl = []
        ind_init = []
        ind_logic = []

        for ind in indicators:
            if ind.upper() == "RSI":
                ind_decl.append("        private Dictionary<Symbol, RelativeStrengthIndex> _rsi = new();")
                ind_init.append("                _rsi[symbol] = algorithm.RSI(symbol, 14, Resolution.Daily);")
                ind_logic.append("                var rsi = _rsi[symbol].Current.Value;")
                ind_logic.append("                if (rsi < 30) direction = InsightDirection.Up;")
                ind_logic.append("                if (rsi > 70) direction = InsightDirection.Down;")
            elif ind.upper() == "EMA":
                ind_decl.append("        private Dictionary<Symbol, ExponentialMovingAverage> _emaFast = new();")
                ind_decl.append("        private Dictionary<Symbol, ExponentialMovingAverage> _emaSlow = new();")
                ind_init.append("                _emaFast[symbol] = algorithm.EMA(symbol, 12, Resolution.Daily);")
                ind_init.append("                _emaSlow[symbol] = algorithm.EMA(symbol, 26, Resolution.Daily);")
                ind_logic.append("                if (_emaFast[symbol] > _emaSlow[symbol]) direction = InsightDirection.Up;")
            elif ind.upper() == "MACD":
                ind_decl.append("        private Dictionary<Symbol, MovingAverageConvergenceDivergence> _macd = new();")
                ind_init.append("                _macd[symbol] = algorithm.MACD(symbol, 12, 26, 9, Resolution.Daily);")
                ind_logic.append("                if (_macd[symbol].Histogram > 0) direction = InsightDirection.Up;")
            elif ind.upper() == "BB":
                ind_decl.append("        private Dictionary<Symbol, BollingerBands> _bb = new();")
                ind_init.append("                _bb[symbol] = algorithm.BB(symbol, 20, 2, Resolution.Daily);")
                ind_logic.append("                var bb = _bb[symbol];")
                ind_logic.append("                if (securities[symbol].Price < bb.LowerBand) direction = InsightDirection.Up;")
                ind_logic.append("                if (securities[symbol].Price > bb.UpperBand) direction = InsightDirection.Down;")

        return f"""using QuantConnect.Algorithm.Framework.Alphas;
using QuantConnect.Data;
using QuantConnect.Indicators;
using System.Collections.Generic;

namespace QuantConnect.Algorithm.CSharp
{{
    public class {strategy_name}AlphaModel : AlphaModel
    {{
{chr(10).join(ind_decl)}

        public override void OnSecuritiesChanged(QCAlgorithm algorithm, SecurityChanges changes)
        {{
            foreach (var security in changes.AddedSecurities)
            {{
                var symbol = security.Symbol;
{chr(10).join(ind_init)}
            }}
        }}

        public override IEnumerable<Insight> Update(QCAlgorithm algorithm, Slice data)
        {{
            var insights = new List<Insight>();
            var securities = algorithm.Securities;

            foreach (var kvp in securities)
            {{
                var symbol = kvp.Key;
                var direction = InsightDirection.Flat;
{chr(10).join(ind_logic)}
                if (direction != InsightDirection.Flat)
                    insights.Add(Insight.Price(symbol, TimeSpan.FromDays(1), direction));
            }}

            return insights;
        }}
    }}
}}"""


class UniverseSelectionGenerator:
    """Generates LEAN CoarseFundamental universe selection C#."""

    @staticmethod
    def generate(symbols: List[str], min_volume: int = 1000000) -> str:
        symbol_list = ", ".join(f'"{s}"' for s in symbols)
        return f"""using QuantConnect.Data.UniverseSelection;
using System.Collections.Generic;
using System.Linq;

namespace QuantConnect.Algorithm.CSharp
{{
    public class StaticUniverse
    {{
        private static readonly string[] Symbols = {{ {symbol_list} }};

        public static IEnumerable<Symbol> SelectSymbols(QCAlgorithm algorithm)
        {{
            return Symbols.Select(s => QuantConnect.Symbol.Create(s, SecurityType.Equity, Market.USA));
        }}
    }}
}}"""


class RiskManagementGenerator:
    """Generates LEAN RiskManagementModel C# from RiskManagerConfig."""

    @staticmethod
    def generate(max_drawdown: float = 0.10, max_position_pct: float = 0.20) -> str:
        return f"""using QuantConnect.Algorithm.Framework.Risk;
using QuantConnect.Algorithm.Framework.Portfolio;
using System.Collections.Generic;

namespace QuantConnect.Algorithm.CSharp
{{
    public class CustomRiskModel : RiskManagementModel
    {{
        private readonly decimal _maxDrawdown = {max_drawdown}m;
        private readonly decimal _maxPositionPct = {max_position_pct}m;
        private decimal _peakValue = 0;

        public override IEnumerable<IPortfolioTarget> ManageRisk(
            QCAlgorithm algorithm, IPortfolioTarget[] targets)
        {{
            var value = algorithm.Portfolio.TotalPortfolioValue;
            if (value > _peakValue) _peakValue = value;

            var drawdown = (_peakValue - value) / _peakValue;
            if (drawdown > _maxDrawdown)
            {{
                // Liquidate all on max drawdown
                foreach (var kvp in algorithm.Portfolio)
                {{
                    if (kvp.Value.Invested)
                        yield return new PortfolioTarget(kvp.Key, 0);
                }}
                yield break;
            }}

            foreach (var target in targets)
            {{
                var pct = target.Quantity * algorithm.Securities[target.Symbol].Price / value;
                if (System.Math.Abs(pct) > _maxPositionPct)
                {{
                    var maxShares = (int)(_maxPositionPct * value / algorithm.Securities[target.Symbol].Price);
                    yield return new PortfolioTarget(target.Symbol, maxShares * System.Math.Sign(target.Quantity));
                }}
                else
                {{
                    yield return target;
                }}
            }}
        }}
    }}
}}"""


class PortfolioConstructionGenerator:
    """Generates LEAN PortfolioConstructionModel C#."""

    @staticmethod
    def generate(method: str = "equal") -> str:
        if method == "momentum":
            # Weight proportional to each insight's magnitude (confidence score).
            pre_loop = (
                "var totalMag = activeInsights\n"
                "                .Sum(i => i.Magnitude.HasValue ? (double)i.Magnitude.Value : 1.0);\n"
                "            if (totalMag <= 0) totalMag = activeInsights.Count;"
            )
            weight_expr = (
                "i.Magnitude.HasValue && totalMag > 0\n"
                "                    ? (double)i.Magnitude.Value / totalMag\n"
                "                    : 1.0 / activeInsights.Count"
            )
        elif method == "risk_parity":
            # Equal-risk contribution: weight inversely proportional to
            # insight magnitude used as a volatility proxy.
            pre_loop = (
                "var invMags = activeInsights\n"
                "                .Select(i => i.Magnitude.HasValue && i.Magnitude.Value > 0\n"
                "                    ? 1.0 / (double)i.Magnitude.Value : 1.0).ToList();\n"
                "            var invTotal = invMags.Sum();\n"
                "            if (invTotal <= 0) invTotal = activeInsights.Count;"
            )
            weight_expr = (
                "invTotal > 0\n"
                "                    ? invMags[activeInsights.IndexOf(i)] / invTotal\n"
                "                    : 1.0 / activeInsights.Count"
            )
        else:
            # Equal weight (default, covers "equal" and unknown values)
            pre_loop = ""
            weight_expr = "1.0 / activeInsights.Count"

        return f"""using QuantConnect.Algorithm.Framework.Portfolio;
using QuantConnect.Algorithm.Framework.Alphas;
using System.Collections.Generic;
using System.Linq;

namespace QuantConnect.Algorithm.CSharp
{{
    public class CustomPortfolioModel : EqualWeightingPortfolioConstructionModel
    {{
        public CustomPortfolioModel() : base(Resolution.Daily) {{ }}

        protected override Dictionary<Insight, double> DetermineTargetPercent(
            List<Insight> activeInsights)
        {{
            var result = new Dictionary<Insight, double>();
            if (activeInsights.Count == 0) return result;

            {pre_loop}
            foreach (var i in activeInsights)
            {{
                var w = {weight_expr};
                result[i] = i.Direction == InsightDirection.Up ? w : -w;
            }}
            return result;
        }}
    }}
}}"""


class LEANProjectGenerator:
    """Generate a complete LEAN C# project."""

    def generate_project(
        self,
        name: str,
        symbols: List[str],
        output_dir: str,
        indicators: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-01-01",
        initial_capital: int = 100000,
        max_drawdown: float = 0.10,
    ) -> str:
        """Generate a complete LEAN project directory.

        Creates: Main.cs, Alpha.cs, Universe.cs, Risk.cs, config.json

        Returns:
            Path to the generated project directory
        """
        indicators = indicators or ["RSI", "EMA"]
        os.makedirs(output_dir, exist_ok=True)

        # Main.cs
        main_cs = f"""using QuantConnect;
using QuantConnect.Algorithm;
using QuantConnect.Algorithm.Framework.Alphas;
using QuantConnect.Algorithm.Framework.Execution;
using QuantConnect.Algorithm.Framework.Portfolio;
using QuantConnect.Algorithm.Framework.Risk;
using QuantConnect.Algorithm.Framework.Selection;
using System;

namespace QuantConnect.Algorithm.CSharp
{{
    public class {name}Algorithm : QCAlgorithm
    {{
        public override void Initialize()
        {{
            SetStartDate({start_date.replace("-", ", ")});
            SetEndDate({end_date.replace("-", ", ")});
            SetCash({initial_capital});

            // Universe
            foreach (var symbol in StaticUniverse.SelectSymbols(this))
                AddEquity(symbol.Value, Resolution.Daily);

            // Framework components
            SetAlpha(new {name}AlphaModel());
            SetPortfolioConstruction(new CustomPortfolioModel());
            SetRiskManagement(new CustomRiskModel());
            SetExecution(new ImmediateExecutionModel());

            // Warm up indicators
            SetWarmUp(TimeSpan.FromDays(60));
        }}
    }}
}}"""
        self._write(output_dir, "Main.cs", main_cs)

        # Alpha.cs
        alpha_cs = AlphaModelGenerator.generate(name, indicators)
        self._write(output_dir, "Alpha.cs", alpha_cs)

        # Universe.cs
        universe_cs = UniverseSelectionGenerator.generate(symbols)
        self._write(output_dir, "Universe.cs", universe_cs)

        # Risk.cs
        risk_cs = RiskManagementGenerator.generate(max_drawdown)
        self._write(output_dir, "Risk.cs", risk_cs)

        # Portfolio.cs
        portfolio_cs = PortfolioConstructionGenerator.generate("equal")
        self._write(output_dir, "Portfolio.cs", portfolio_cs)

        # config.json
        config = {
            "algorithm-type-name": f"{name}Algorithm",
            "algorithm-language": "CSharp",
            "parameters": {},
            "description": f"Generated by stocks_plugin LEANProjectGenerator",
        }
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Generated LEAN project: %s (%d files)", output_dir, 6)
        return output_dir

    @staticmethod
    def _write(directory: str, filename: str, content: str) -> None:
        path = os.path.join(directory, filename)
        with open(path, "w") as f:
            f.write(content)
        logger.debug("  Written: %s", path)
