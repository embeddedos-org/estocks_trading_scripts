"""
RL Trading Agent using Stable-Baselines3
============================================

Wrapper around SB3 algorithms (PPO, A2C, SAC) for training and
deploying RL trading agents. Integrates with BacktestEngineV2
for evaluation and Optuna for hyperparameter optimization.

Requires: pip install stable-baselines3 gymnasium

Usage:
    trader = RLTrader(algorithm="PPO")
    trader.train(df_spy, total_timesteps=100_000)
    result = trader.backtest(df_test)
    print(f"RL Sharpe: {result.sharpe_ratio}")
    trader.save_model("models/ppo_trader.zip")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO, A2C, SAC  # type: ignore[import-untyped]
    from stable_baselines3.common.callbacks import EvalCallback  # type: ignore[import-untyped]
    from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore[import-untyped]
    _HAS_SB3 = True
except ImportError:
    _HAS_SB3 = False
    logger.debug("stable-baselines3 not installed — RL agent unavailable")

try:
    import optuna  # type: ignore[import-untyped]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.ml.rl_trading_env import TradingEnv


def _require_sb3() -> None:
    if not _HAS_SB3:
        raise ImportError(
            "stable-baselines3 is required for the RL agent. "
            "Install with: pip install stable-baselines3>=2.0"
        )


_ALGORITHMS: Dict[str, Any] = {}
if _HAS_SB3:
    _ALGORITHMS = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }


class RLTrader:
    """RL-based trading agent using stable-baselines3.

    Trains on historical OHLCV data via a custom Gymnasium environment
    and produces trading signals compatible with BacktestEngineV2.

    Args:
        algorithm: RL algorithm name — "PPO", "A2C", or "SAC".
        env_config: Dict with TradingEnv parameters:
            - reward_type: "pnl" | "sharpe" | "sortino" | "risk_adjusted"
            - commission: float (default 0.001)
            - slippage_pct: float (default 0.0005)
            - max_drawdown_threshold: float (default 0.20)
            - initial_capital: float (default 100000)
        model_kwargs: Additional keyword arguments for the SB3 model.
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        env_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        _require_sb3()

        self._algorithm_name = algorithm.upper()
        if self._algorithm_name not in _ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Choose from: {list(_ALGORITHMS.keys())}"
            )

        self._algo_class = _ALGORITHMS[self._algorithm_name]
        self._env_config = env_config or {}
        self._model_kwargs = model_kwargs or {}
        self._model: Any = None
        self._train_env: Any = None
        self._is_trained = False

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 100_000,
        eval_df: Optional[pd.DataFrame] = None,
        eval_freq: int = 5000,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """Train the RL agent on historical OHLCV data.

        Args:
            df: Training OHLCV DataFrame.
            total_timesteps: Total training timesteps.
            eval_df: Optional evaluation DataFrame for monitoring.
            eval_freq: Evaluate every N timesteps.
            verbose: Verbosity level (0=silent, 1=info).

        Returns:
            Dict with training metrics: mean_reward, n_episodes, etc.
        """
        logger.info(
            "Training %s agent: %d timesteps, reward=%s",
            self._algorithm_name,
            total_timesteps,
            self._env_config.get("reward_type", "pnl"),
        )

        # Create training environment
        self._train_env = DummyVecEnv([lambda: TradingEnv(df, **self._env_config)])

        # Build model
        default_kwargs = {"verbose": verbose}
        if self._algorithm_name in ("PPO", "A2C"):
            default_kwargs.update({
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "gamma": 0.99,
                "ent_coef": 0.01,
            })
        elif self._algorithm_name == "SAC":
            default_kwargs.update({
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "buffer_size": 50_000,
            })

        default_kwargs.update(self._model_kwargs)

        self._model = self._algo_class(
            "MlpPolicy",
            self._train_env,
            **default_kwargs,
        )

        # Evaluation callback
        callbacks = []
        if eval_df is not None:
            eval_env = DummyVecEnv([lambda: TradingEnv(eval_df, **self._env_config)])
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=1,
                verbose=0,
            )
            callbacks.append(eval_callback)

        # Train
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
        )

        self._is_trained = True

        # Collect metrics
        metrics = {
            "algorithm": self._algorithm_name,
            "total_timesteps": total_timesteps,
            "reward_type": self._env_config.get("reward_type", "pnl"),
        }

        logger.info("Training complete: %s", metrics)
        return metrics

    def predict(self, df: pd.DataFrame) -> int:
        """Predict trading action for current market state.

        Args:
            df: OHLCV DataFrame (uses latest bar's features).

        Returns:
            Action signal: -1 (SELL), 0 (HOLD), or +1 (BUY).
        """
        self._check_trained()

        env = TradingEnv(df, **self._env_config)
        obs, _ = env.reset()

        # Step to the last bar
        for i in range(len(df) - 2):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

        # Predict on final state
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action) - 1  # Map 0,1,2 → -1,0,+1

    def backtest(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
    ) -> Any:
        """Run trained agent through BacktestEngineV2.

        Args:
            df: OHLCV DataFrame for backtesting.
            initial_capital: Starting capital.

        Returns:
            BacktestResultV2 with full metrics.
        """
        self._check_trained()

        from shared.backtesting.backtest_engine_v2 import BacktestEngineV2

        env = TradingEnv(df, **self._env_config)
        obs, _ = env.reset()

        actions: List[int] = []
        for _ in range(len(df) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            action_int = int(action)
            actions.append(action_int)
            obs, _, terminated, truncated, _ = env.step(action_int)
            if terminated or truncated:
                break

        # Pad actions to match data length
        while len(actions) < len(df):
            actions.append(1)  # HOLD

        # Convert to strategy function for BacktestEngineV2
        action_map = {0: -1, 1: 0, 2: 1}  # SELL, HOLD, BUY

        def rl_strategy(context: Any) -> Dict[str, int]:
            bar_idx = context.bar_index
            if bar_idx < len(actions):
                signal = action_map.get(actions[bar_idx], 0)
            else:
                signal = 0
            symbols = list(context.bars.keys())
            return {sym: signal for sym in symbols}

        engine = BacktestEngineV2(initial_capital=initial_capital)
        engine.load_data(df)
        result = engine.run(rl_strategy)

        logger.info(
            "RL backtest: Return=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%, Trades=%d",
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
            result.total_trades,
        )

        return result

    def optimize_hyperparams(
        self,
        df: pd.DataFrame,
        n_trials: int = 50,
        eval_df: Optional[pd.DataFrame] = None,
        timesteps_per_trial: int = 50_000,
    ) -> Dict[str, Any]:
        """Optimize RL hyperparameters using Optuna.

        Tunes: learning_rate, n_steps, gamma, ent_coef, clip_range (for PPO).

        Args:
            df: Training OHLCV DataFrame.
            n_trials: Number of Optuna trials.
            eval_df: Evaluation DataFrame (uses df if not provided).
            timesteps_per_trial: Timesteps per trial.

        Returns:
            Dict with best parameters and performance metrics.
        """
        if not _HAS_OPTUNA:
            raise ImportError(
                "optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )

        test_df = eval_df if eval_df is not None else df

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            gamma = trial.suggest_float("gamma", 0.9, 0.9999)

            kwargs: Dict[str, Any] = {
                "learning_rate": lr,
                "gamma": gamma,
            }

            if self._algorithm_name in ("PPO", "A2C"):
                kwargs["n_steps"] = trial.suggest_int("n_steps", 256, 4096, step=256)
                kwargs["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)

            if self._algorithm_name == "PPO":
                kwargs["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)

            try:
                env = DummyVecEnv([lambda: TradingEnv(df, **self._env_config)])
                model = self._algo_class("MlpPolicy", env, verbose=0, **kwargs)
                model.learn(total_timesteps=timesteps_per_trial)

                # Evaluate
                eval_env = TradingEnv(test_df, **self._env_config)
                obs, _ = eval_env.reset()
                total_reward = 0.0
                for _ in range(len(test_df) - 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                    total_reward += reward
                    if terminated or truncated:
                        break

                return total_reward

            except Exception as e:
                logger.warning("Optuna trial failed: %s", e)
                return float("-inf")

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info(
            "RL hyperparameter optimization complete (%d trials). Best params: %s",
            n_trials, best_params,
        )

        return {
            "best_params": best_params,
            "best_value": study.best_value,
            "n_trials": n_trials,
            "algorithm": self._algorithm_name,
        }

    def save_model(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path (e.g. "models/ppo_trader.zip").
        """
        self._check_trained()
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(filepath))
        logger.info("RL model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to load.
        """
        self._model = self._algo_class.load(path)
        self._is_trained = True
        logger.info("RL model loaded from %s", path)

    def _check_trained(self) -> None:
        """Raise if model has not been trained."""
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "Model not trained. Call train() or load_model() first."
            )
