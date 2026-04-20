"""
Tests for shared/ml/rl_agent.py and shared/ml/rl_trading_env.py
=================================================================

Covers:
- RLTradingEnv: reset(), step(), reward calculation
- RLAgent: train(), predict(), save/load
- Action mapping: 0=SELL, 1=HOLD, 2=BUY
- Fallback when stable-baselines3 missing
"""
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_HAS_GYM = False
try:
    import gymnasium
    _HAS_GYM = True
except ImportError:
    pass

from shared.ml.rl_agent import _HAS_SB3


def _make_ohlcv(n=200, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100.0
    rows = []
    for i in range(n):
        price *= 1 + rng.randn() * 0.015
        rows.append({
            "date": dates[i],
            "open": price * 1.001,
            "high": price * 1.005,
            "low": price * 0.995,
            "close": price,
            "volume": 1_000_000,
        })
    return pd.DataFrame(rows)


# ─── TradingEnv Tests ───

@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
class TestTradingEnvInit:
    @pytest.fixture
    def env(self):
        from shared.ml.rl_trading_env import TradingEnv
        return TradingEnv(_make_ohlcv(200))

    def test_action_space(self, env):
        assert env.action_space.n == 3

    def test_observation_space_shape(self, env):
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1
        assert obs_shape[0] == env.n_features

    def test_n_features_includes_state(self, env):
        assert env._n_state_features == 3
        assert env._n_features == env._n_market_features + 3


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
class TestTradingEnvReset:
    @pytest.fixture
    def env(self):
        from shared.ml.rl_trading_env import TradingEnv
        return TradingEnv(_make_ohlcv(200))

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape_matches_space(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape

    def test_reset_portfolio_value(self, env):
        env.reset()
        assert env._portfolio_value == env._initial_capital

    def test_reset_position_flat(self, env):
        env.reset()
        assert env._position == 0
        assert env._shares == 0


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
class TestTradingEnvStep:
    @pytest.fixture
    def env(self):
        from shared.ml.rl_trading_env import TradingEnv
        return TradingEnv(_make_ohlcv(200))

    def test_step_returns_five_values(self, env):
        env.reset()
        result = env.step(1)
        assert len(result) == 5

    def test_hold_action_no_position(self, env):
        env.reset()
        env.step(1)
        assert env._position == 0

    def test_buy_action_opens_long(self, env):
        env.reset()
        env.step(2)
        assert env._position == 1
        assert env._shares > 0

    def test_sell_action_opens_short(self, env):
        env.reset()
        env.step(0)
        assert env._position == -1
        assert env._shares > 0

    def test_action_mapping(self, env):
        from shared.ml.rl_trading_env import TradingEnv
        assert TradingEnv.SELL == 0
        assert TradingEnv.HOLD == 1
        assert TradingEnv.BUY == 2


@pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
class TestTradingEnvReward:
    def test_pnl_reward_is_float(self):
        from shared.ml.rl_trading_env import TradingEnv
        env = TradingEnv(_make_ohlcv(200), reward_type="pnl")
        env.reset()
        _, reward, _, _, _ = env.step(2)
        assert isinstance(reward, float)

    def test_sharpe_reward_early_steps(self):
        from shared.ml.rl_trading_env import TradingEnv
        env = TradingEnv(_make_ohlcv(200), reward_type="sharpe")
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert isinstance(reward, float)

    def test_risk_adjusted_reward(self):
        from shared.ml.rl_trading_env import TradingEnv
        env = TradingEnv(_make_ohlcv(200), reward_type="risk_adjusted")
        env.reset()
        _, reward, _, _, _ = env.step(2)
        assert isinstance(reward, float)

    def test_episode_terminates_at_end(self):
        from shared.ml.rl_trading_env import TradingEnv
        df = _make_ohlcv(50)
        env = TradingEnv(df)
        env.reset()
        terminated = False
        for _ in range(100):
            _, _, terminated, _, _ = env.step(1)
            if terminated:
                break
        assert terminated


# ─── RLTrader Tests ───

class TestRLTraderInit:
    @pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
    def test_init_ppo(self):
        from shared.ml.rl_agent import RLTrader
        trader = RLTrader(algorithm="PPO")
        assert trader._algorithm_name == "PPO"

    @pytest.mark.skipif(_HAS_SB3, reason="stable-baselines3 IS installed")
    def test_init_requires_sb3(self):
        with pytest.raises(ImportError):
            from shared.ml.rl_agent import RLTrader
            RLTrader()

    @pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
    def test_invalid_algorithm(self):
        from shared.ml.rl_agent import RLTrader
        with pytest.raises(ValueError, match="Unsupported"):
            RLTrader(algorithm="INVALID")


class TestRLTraderPredict:
    @pytest.mark.skipif(not _HAS_SB3, reason="stable-baselines3 not installed")
    def test_predict_before_train_raises(self):
        from shared.ml.rl_agent import RLTrader
        trader = RLTrader()
        with pytest.raises(RuntimeError, match="not trained"):
            trader.predict(_make_ohlcv(100))

    @pytest.mark.skipif(not _HAS_SB3 or not _HAS_GYM, reason="sb3/gymnasium not installed")
    def test_predict_returns_mapped_action(self):
        from shared.ml.rl_agent import RLTrader
        trader = RLTrader()
        trader._is_trained = True
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([1]), None)
        trader._model = mock_model
        result = trader.predict(_make_ohlcv(50))
        assert result in (-1, 0, 1)


class TestRLTraderActionMapping:
    @pytest.mark.skipif(not _HAS_GYM, reason="gymnasium not installed")
    def test_action_values(self):
        from shared.ml.rl_trading_env import TradingEnv
        assert TradingEnv.SELL == 0
        assert TradingEnv.HOLD == 1
        assert TradingEnv.BUY == 2


class TestFallbackWhenSB3Missing:
    def test_has_sb3_flag_exists(self):
        assert isinstance(_HAS_SB3, bool)

    @pytest.mark.skipif(_HAS_SB3, reason="stable-baselines3 IS installed")
    def test_require_sb3_raises_when_missing(self):
        from shared.ml.rl_agent import _require_sb3
        with pytest.raises(ImportError):
            _require_sb3()
