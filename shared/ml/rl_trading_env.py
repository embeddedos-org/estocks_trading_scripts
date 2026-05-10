"""
Custom Gymnasium Trading Environment
========================================

FinRL-compatible trading environment for reinforcement learning.
Observation space uses 30+ features from MLRegimeClassifier plus
position state. Supports configurable reward functions.

Requires: pip install gymnasium

Usage:
    env = TradingEnv(df, reward_type="sharpe")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym  # type: ignore[import-untyped]
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    logger.debug("gymnasium not installed — RL environment unavailable")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.ml.regime_classifier import MLRegimeClassifier


def _require_gymnasium() -> None:
    if not _HAS_GYM:
        raise ImportError(
            "gymnasium is required for the RL trading environment. "
            "Install with: pip install gymnasium>=0.29"
        )


class TradingEnv(gym.Env if _HAS_GYM else object):  # type: ignore[misc]
    """Custom Gymnasium environment for RL-based trading.

    Walks through historical OHLCV data bar-by-bar. The agent observes
    normalized features and decides to BUY, HOLD, or SELL.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        initial_capital: Starting portfolio value.
        commission: Commission rate as fraction (e.g. 0.001 = 0.1%).
        slippage_pct: Slippage as fraction of price (e.g. 0.0005).
        reward_type: Reward function — "pnl", "sharpe", "sortino", or "risk_adjusted".
        max_drawdown_threshold: Episode terminates if drawdown exceeds this.
        window_size: Rolling window for Sharpe/Sortino reward (bars).
    """

    metadata = {"render_modes": ["human"]}

    # Actions
    SELL = 0
    HOLD = 1
    BUY = 2

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage_pct: float = 0.0005,
        reward_type: str = "pnl",
        max_drawdown_threshold: float = 0.20,
        window_size: int = 20,
    ) -> None:
        _require_gymnasium()
        super().__init__()

        self._raw_df = df.copy()
        self._raw_df.columns = [c.strip().lower() for c in self._raw_df.columns]
        self._initial_capital = initial_capital
        self._commission = commission
        self._slippage_pct = slippage_pct
        self._reward_type = reward_type
        self._max_dd_threshold = max_drawdown_threshold
        self._window_size = window_size

        # Compute features
        self._features_df = MLRegimeClassifier.compute_features(self._raw_df)
        self._features_df = self._features_df.fillna(0)

        # Normalize features
        feat_mean = self._features_df.mean()
        feat_std = self._features_df.std().replace(0, 1)
        self._features_normalized = (self._features_df - feat_mean) / feat_std
        self._features_normalized = self._features_normalized.clip(-5, 5)

        self._n_market_features = len(self._features_normalized.columns)

        # Extra state features: position (-1/0/+1), unrealized_pnl, portfolio_value_ratio
        self._n_state_features = 3
        self._n_features = self._n_market_features + self._n_state_features

        # Spaces
        self.action_space = spaces.Discrete(3)  # SELL=0, HOLD=1, BUY=2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._n_features,),
            dtype=np.float32,
        )

        # State
        self._current_step = 0
        self._capital = initial_capital
        self._position = 0  # -1, 0, or +1
        self._shares = 0
        self._entry_price = 0.0
        self._portfolio_value = initial_capital
        self._peak_value = initial_capital
        self._equity_curve: List[float] = []
        self._returns: List[float] = []

        # Valid range (skip NaN rows at start)
        first_valid = self._features_normalized.notna().all(axis=1).idxmax()
        self._start_idx = max(0, int(first_valid)) if isinstance(first_valid, (int, np.integer)) else 0
        if hasattr(first_valid, 'item'):
            self._start_idx = int(self._raw_df.index.get_loc(first_valid))
        self._max_steps = len(self._raw_df) - 1

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)

        self._current_step = self._start_idx
        self._capital = self._initial_capital
        self._position = 0
        self._shares = 0
        self._entry_price = 0.0
        self._portfolio_value = self._initial_capital
        self._peak_value = self._initial_capital
        self._equity_curve = [self._initial_capital]
        self._returns = []

        obs = self._get_observation()
        info = {"portfolio_value": self._portfolio_value, "position": self._position}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: 0=SELL, 1=HOLD, 2=BUY.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        prev_value = self._portfolio_value
        current_price = float(self._raw_df.iloc[self._current_step]["close"])

        # Execute action
        self._execute_action(action, current_price)

        # Advance step
        self._current_step += 1

        # Update portfolio value
        new_price = float(self._raw_df.iloc[self._current_step]["close"])
        self._update_portfolio_value(new_price)

        # Track
        self._equity_curve.append(self._portfolio_value)
        ret = (self._portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        self._returns.append(ret)

        if self._portfolio_value > self._peak_value:
            self._peak_value = self._portfolio_value

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value if self._peak_value > 0 else 0
        terminated = (
            self._current_step >= self._max_steps
            or self._portfolio_value <= 0
            or drawdown > self._max_dd_threshold
        )
        truncated = False

        obs = self._get_observation()
        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "drawdown": drawdown,
            "total_return": (self._portfolio_value - self._initial_capital) / self._initial_capital,
            "step": self._current_step,
        }

        return obs, float(reward), terminated, truncated, info

    def _execute_action(self, action: int, price: float) -> None:
        """Execute a trading action."""
        target_position = action - 1  # SELL=-1, HOLD=0, BUY=+1

        if target_position == self._position:
            return

        # Close existing position
        if self._position != 0:
            if self._position == 1:  # close long
                exit_price = price * (1 - self._slippage_pct)
                proceeds = self._shares * exit_price
                comm = proceeds * self._commission
                self._capital += proceeds - comm
            elif self._position == -1:  # close short
                exit_price = price * (1 + self._slippage_pct)
                cost = self._shares * exit_price
                comm = cost * self._commission
                self._capital -= cost + comm

            self._position = 0
            self._shares = 0
            self._entry_price = 0.0

        # Open new position
        if target_position != 0:
            if target_position == 1:  # go long
                entry_price = price * (1 + self._slippage_pct)
                shares = int(self._capital * 0.95 / entry_price) if entry_price > 0 else 0
                if shares > 0:
                    cost = shares * entry_price
                    comm = cost * self._commission
                    self._capital -= cost + comm
                    self._position = 1
                    self._shares = shares
                    self._entry_price = entry_price

            elif target_position == -1:  # go short
                entry_price = price * (1 - self._slippage_pct)
                shares = int(self._capital * 0.95 / entry_price) if entry_price > 0 else 0
                if shares > 0:
                    proceeds = shares * entry_price
                    comm = proceeds * self._commission
                    self._capital += proceeds - comm
                    self._position = -1
                    self._shares = shares
                    self._entry_price = entry_price

    def _update_portfolio_value(self, price: float) -> None:
        """Recalculate portfolio value based on current price."""
        if self._position == 1:
            self._portfolio_value = self._capital + self._shares * price
        elif self._position == -1:
            self._portfolio_value = self._capital - self._shares * price
        else:
            self._portfolio_value = self._capital

    def _get_observation(self) -> np.ndarray:
        """Build observation vector: market features + state features."""
        market_features = self._features_normalized.iloc[self._current_step].values.astype(np.float32)

        current_price = float(self._raw_df.iloc[self._current_step]["close"])
        unrealized_pnl = 0.0
        if self._position == 1 and self._entry_price > 0:
            unrealized_pnl = (current_price - self._entry_price) / self._entry_price
        elif self._position == -1 and self._entry_price > 0:
            unrealized_pnl = (self._entry_price - current_price) / self._entry_price

        value_ratio = self._portfolio_value / self._initial_capital - 1.0

        state_features = np.array(
            [float(self._position), unrealized_pnl, value_ratio],
            dtype=np.float32,
        )

        return np.concatenate([market_features, state_features])

    def _compute_reward(self) -> float:
        """Compute reward based on configured reward type."""
        if self._reward_type == "pnl":
            if len(self._returns) == 0:
                return 0.0
            return self._returns[-1] * 100

        elif self._reward_type == "sharpe":
            if len(self._returns) < self._window_size:
                return self._returns[-1] * 100 if self._returns else 0.0
            window = np.array(self._returns[-self._window_size:])
            std = np.std(window, ddof=1)
            return float(np.mean(window) / std * np.sqrt(252)) if std > 0 else 0.0

        elif self._reward_type == "sortino":
            if len(self._returns) < self._window_size:
                return self._returns[-1] * 100 if self._returns else 0.0
            window = np.array(self._returns[-self._window_size:])
            downside = window[window < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
            return float(np.mean(window) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        elif self._reward_type == "risk_adjusted":
            if len(self._returns) == 0:
                return 0.0
            pnl_reward = self._returns[-1] * 100
            drawdown = (self._peak_value - self._portfolio_value) / self._peak_value if self._peak_value > 0 else 0
            return pnl_reward - drawdown * 10

        return 0.0

    @property
    def n_features(self) -> int:
        """Total number of observation features."""
        return self._n_features
