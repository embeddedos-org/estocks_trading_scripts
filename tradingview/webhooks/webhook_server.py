"""
TradingView Webhook Server
FastAPI server that receives TradingView alert webhooks, validates them,
and routes orders to configured broker adapters.

stocks_plugin - tradingview/webhooks/webhook_server.py
"""

import hashlib
import hmac
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Graceful imports from shared modules ───
try:
    from shared.config import Config
except ImportError:
    Config = None

try:
    from shared.notifier import AlertDispatcher
except ImportError:
    AlertDispatcher = None

# ─── Logging Setup ───
logger = logging.getLogger("webhook_server")


# ─── Configuration Loading ───
def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return _default_config()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config: {e}")
        return _default_config()


def _default_config() -> dict:
    """Return default configuration."""
    return {
        "server": {"host": "0.0.0.0", "port": 5000, "debug": False},
        "security": {
            "hmac_secret": "default_secret",
            "hmac_algorithm": "sha256",
            "allowed_ips": ["127.0.0.1"],
            "require_passphrase": False,
            "passphrase": "",
        },
        "rate_limiting": {
            "enabled": True,
            "max_requests_per_minute": 60,
            "window_seconds": 60,
        },
        "broker_routing": {"default_broker": "interactive_brokers", "routes": []},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    }


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handlers = []

    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

    log_file = log_config.get("file")
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_bytes", 10485760),
            backupCount=log_config.get("backup_count", 5),
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)


# ─── Pydantic Models ───
class AlertPayload(BaseModel):
    """TradingView alert webhook payload."""
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL)")
    action: str = Field(..., description="Trade action: buy, sell, close")
    price: float = Field(..., ge=0, description="Current price")
    quantity: Optional[float] = Field(None, ge=0, description="Order quantity")
    order_type: str = Field("market", description="Order type: market, limit, stop")
    passphrase: Optional[str] = Field(None, description="Webhook passphrase for auth")
    timestamp: Optional[str] = Field(None, description="Alert timestamp")
    strategy: Optional[str] = Field(None, description="Strategy name")
    timeframe: Optional[str] = Field(None, description="Chart timeframe")
    message: Optional[str] = Field(None, description="Additional message")
    regime: Optional[str] = Field(None, description="Market regime: TRENDING, RANGING, VOLATILE")
    signal: Optional[str] = Field(None, description="Signal type from strategy")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    last_alert_time: Optional[str]
    total_alerts_processed: int
    version: str


class OrderResult(BaseModel):
    """Result of order execution attempt."""
    success: bool
    broker: str
    order_id: Optional[str] = None
    message: str
    timestamp: str


# ─── Broker Adapter Pattern ───
class BrokerAdapter(ABC):
    """Abstract base class for broker adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        """Place an order with the broker."""
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker adapter name."""
        ...


class IBBrokerAdapter(BrokerAdapter):
    """Interactive Brokers adapter (placeholder implementation)."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._connected = False
        self._order_counter = 0

    @property
    def name(self) -> str:
        return "interactive_brokers"

    def connect(self) -> bool:
        logger.info("Connecting to Interactive Brokers...")
        self._connected = True
        return True

    def disconnect(self) -> None:
        logger.info("Disconnecting from Interactive Brokers")
        self._connected = False

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        if not self._connected:
            self.connect()

        self._order_counter += 1
        order_id = f"IB-{self._order_counter:06d}"
        logger.info(
            f"[IB] Order {order_id}: {action} {quantity} {symbol} @ {price} ({order_type})"
        )

        return OrderResult(
            success=True,
            broker=self.name,
            order_id=order_id,
            message=f"Order placed: {action} {quantity} {symbol} @ {price}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_account_info(self) -> dict:
        return {
            "broker": self.name,
            "connected": self._connected,
            "account_type": "paper",
            "buying_power": 100000.0,
        }


class TradeStationBrokerAdapter(BrokerAdapter):
    """TradeStation adapter (placeholder implementation)."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._connected = False
        self._order_counter = 0

    @property
    def name(self) -> str:
        return "tradestation"

    def connect(self) -> bool:
        logger.info("Connecting to TradeStation...")
        self._connected = True
        return True

    def disconnect(self) -> None:
        logger.info("Disconnecting from TradeStation")
        self._connected = False

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str,
        price: float,
    ) -> OrderResult:
        if not self._connected:
            self.connect()

        self._order_counter += 1
        order_id = f"TS-{self._order_counter:06d}"
        logger.info(
            f"[TS] Order {order_id}: {action} {quantity} {symbol} @ {price} ({order_type})"
        )

        return OrderResult(
            success=True,
            broker=self.name,
            order_id=order_id,
            message=f"Order placed: {action} {quantity} {symbol} @ {price}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_account_info(self) -> dict:
        return {
            "broker": self.name,
            "connected": self._connected,
            "account_type": "paper",
        }


# ─── Rate Limiter ───
class RateLimiter:
    """Per-IP rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from client_ip is within rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds

        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > cutoff
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client_ip."""
        now = time.time()
        cutoff = now - self.window_seconds
        current = len([t for t in self._requests.get(client_ip, []) if t > cutoff])
        return max(0, self.max_requests - current)


# ─── Broker Router ───
class BrokerRouter:
    """Routes symbols to appropriate broker adapters based on config patterns."""

    def __init__(self, config: dict):
        self._routes = config.get("broker_routing", {}).get("routes", [])
        self._default = config.get("broker_routing", {}).get(
            "default_broker", "interactive_brokers"
        )
        self._adapters: dict[str, BrokerAdapter] = {
            "interactive_brokers": IBBrokerAdapter(),
            "tradestation": TradeStationBrokerAdapter(),
        }

    def get_broker(self, symbol: str) -> BrokerAdapter:
        """Get the appropriate broker adapter for a symbol."""
        for route in self._routes:
            pattern = route.get("pattern", "")
            broker_name = route.get("broker", self._default)
            if re.match(pattern, symbol, re.IGNORECASE):
                adapter = self._adapters.get(broker_name)
                if adapter:
                    logger.debug(
                        f"Routing {symbol} to {broker_name} (pattern: {pattern})"
                    )
                    return adapter

        default_adapter = self._adapters.get(self._default)
        if default_adapter:
            return default_adapter

        raise ValueError(f"No broker adapter found for symbol: {symbol}")

    def get_all_brokers(self) -> dict[str, BrokerAdapter]:
        """Return all registered broker adapters."""
        return self._adapters


# ─── HMAC Validation ───
def validate_hmac_signature(
    payload: bytes, signature: str, secret: str, algorithm: str = "sha256"
) -> bool:
    """Validate HMAC signature from X-Webhook-Signature header."""
    if not signature or not secret:
        return False

    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    expected = hmac.new(secret.encode(), payload, hash_func).hexdigest()

    return hmac.compare_digest(expected, signature)


# ─── Application Factory ───
def create_app(config_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_config(config_path)
    setup_logging(config)

    server_config = config.get("server", {})
    app = FastAPI(
        title=server_config.get("title", "TradingView Webhook Server"),
        version=server_config.get("version", "1.0.0"),
        debug=server_config.get("debug", False),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Application State ───
    app.state.config = config
    app.state.start_time = time.time()
    app.state.last_alert_time = None
    app.state.total_alerts = 0
    app.state.broker_router = BrokerRouter(config)

    rate_config = config.get("rate_limiting", {})
    app.state.rate_limiter = RateLimiter(
        max_requests=rate_config.get("max_requests_per_minute", 60),
        window_seconds=rate_config.get("window_seconds", 60),
    )

    # Initialize AlertDispatcher if available
    app.state.alert_dispatcher = None
    if AlertDispatcher is not None:
        try:
            app.state.alert_dispatcher = AlertDispatcher()
            logger.info("AlertDispatcher initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AlertDispatcher: {e}")

    # ─── Logging Middleware ───
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"→ {request.method} {request.url.path} from {client_ip}")

        response = await call_next(request)

        duration = time.time() - start
        logger.info(
            f"← {request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)"
        )
        return response

    # ─── Health Endpoint ───
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint returning server status and uptime."""
        uptime = time.time() - app.state.start_time
        last_alert = (
            app.state.last_alert_time.isoformat()
            if app.state.last_alert_time
            else None
        )

        return HealthResponse(
            status="healthy",
            uptime_seconds=round(uptime, 2),
            last_alert_time=last_alert,
            total_alerts_processed=app.state.total_alerts,
            version=server_config.get("version", "1.0.0"),
        )

    # ─── Webhook Endpoint ───
    @app.post("/webhook")
    async def receive_webhook(request: Request):
        """
        Receive TradingView alert webhook.

        Validates HMAC signature, checks rate limits, authenticates passphrase,
        routes to appropriate broker, and dispatches notifications.
        """
        client_ip = request.client.host if request.client else "unknown"
        security_config = config.get("security", {})

        # ── IP Allowlist Check ──
        allowed_ips = security_config.get("allowed_ips", [])
        if allowed_ips and client_ip not in allowed_ips:
            logger.warning(f"Rejected request from unauthorized IP: {client_ip}")
            raise HTTPException(status_code=403, detail="IP not allowed")

        # ── Rate Limiting ──
        if config.get("rate_limiting", {}).get("enabled", True):
            if not app.state.rate_limiter.is_allowed(client_ip):
                remaining = app.state.rate_limiter.get_remaining(client_ip)
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again later. Remaining: {remaining}",
                )

        # ── Read Raw Body ──
        raw_body = await request.body()

        # ── HMAC Signature Validation ──
        hmac_secret = security_config.get("hmac_secret", "")
        signature = request.headers.get("X-Webhook-Signature", "")

        if hmac_secret and hmac_secret != "CHANGE_ME_TO_A_SECURE_SECRET_KEY":
            algorithm = security_config.get("hmac_algorithm", "sha256")
            if not validate_hmac_signature(raw_body, signature, hmac_secret, algorithm):
                logger.warning(f"Invalid HMAC signature from {client_ip}")
                raise HTTPException(status_code=401, detail="Invalid signature")

        # ── Parse Payload ──
        try:
            import json
            payload_dict = json.loads(raw_body)
            alert = AlertPayload(**payload_dict)
        except Exception as e:
            logger.error(f"Invalid payload from {client_ip}: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid payload: {str(e)}")

        # ── Passphrase Validation ──
        if security_config.get("require_passphrase", False):
            expected_passphrase = security_config.get("passphrase", "")
            if alert.passphrase != expected_passphrase:
                logger.warning(f"Invalid passphrase from {client_ip}")
                raise HTTPException(status_code=401, detail="Invalid passphrase")

        # ── Validate Action ──
        valid_actions = {"buy", "sell", "close", "long", "short"}
        if alert.action.lower() not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {alert.action}. Must be one of: {valid_actions}",
            )

        # ── Route to Broker ──
        try:
            broker = app.state.broker_router.get_broker(alert.symbol)
            quantity = alert.quantity or 1.0

            order_result = broker.place_order(
                symbol=alert.symbol,
                action=alert.action.lower(),
                quantity=quantity,
                order_type=alert.order_type,
                price=alert.price,
            )
        except Exception as e:
            logger.error(f"Broker execution error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Broker execution failed: {str(e)}"
            )

        # ── Update State ──
        app.state.last_alert_time = datetime.now(timezone.utc)
        app.state.total_alerts += 1

        # ── Dispatch Notifications ──
        if app.state.alert_dispatcher:
            try:
                app.state.alert_dispatcher.dispatch(
                    title=f"Trade Alert: {alert.action.upper()} {alert.symbol}",
                    message=(
                        f"Action: {alert.action}\n"
                        f"Symbol: {alert.symbol}\n"
                        f"Price: {alert.price}\n"
                        f"Quantity: {quantity}\n"
                        f"Broker: {broker.name}\n"
                        f"Order ID: {order_result.order_id}"
                    ),
                    level="info",
                )
            except Exception as e:
                logger.warning(f"Notification dispatch failed: {e}")

        logger.info(
            f"Alert processed: {alert.action} {alert.symbol} @ {alert.price} "
            f"via {broker.name} → {order_result.order_id}"
            f"{f' | regime={alert.regime}' if alert.regime else ''}"
            f"{f' | strategy={alert.strategy}' if alert.strategy else ''}"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "alert": {
                    "symbol": alert.symbol,
                    "action": alert.action,
                    "price": alert.price,
                    "quantity": quantity,
                    "regime": alert.regime,
                    "strategy": alert.strategy,
                },
                "order": order_result.model_dump(),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ─── AI-Enhanced Webhook Endpoint ───
    @app.post("/ai-webhook")
    async def receive_ai_webhook(request: Request):
        """
        AI-enhanced TradingView webhook.

        Routes alerts through the SelfLearningAgent before executing.
        The agent can confirm, override, or block the Pine Script signal
        based on ML analysis, news sentiment, and trade history.

        Flow: TradingView Alert → AI Agent.decide() → Broker Execution
        """
        client_ip = request.client.host if request.client else "unknown"
        security_config = config.get("security", {})

        # ── Rate Limiting ──
        if config.get("rate_limiting", {}).get("enabled", True):
            if not app.state.rate_limiter.is_allowed(client_ip):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # ── Parse Payload ──
        raw_body = await request.body()
        try:
            import json
            payload_dict = json.loads(raw_body)
            alert = AlertPayload(**payload_dict)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid payload: {str(e)}")

        # ── Fetch Data for AI Agent ──
        ai_result = {"agent_used": False, "agent_action": None, "confidence": 0}

        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

            from shared.data.public_data_fetcher import PublicDataFetcher
            from shared.ml.self_learning_agent import SelfLearningAgent, AgentConfig

            # Initialize agent (lazy singleton)
            if not hasattr(app.state, "ai_agent") or app.state.ai_agent is None:
                import tempfile
                db_path = os.path.join(tempfile.gettempdir(), "webhook_agent_memory.db")
                app.state.ai_agent = SelfLearningAgent(AgentConfig(db_path=db_path))

            # Fetch market data
            fetcher = PublicDataFetcher(cache_enabled=True)
            df = fetcher.fetch_ohlcv(alert.symbol, period="6mo", interval="1d")

            if df is not None and len(df) >= 60:
                decision = app.state.ai_agent.decide(df, symbol=alert.symbol)
                ai_action = decision.get("action", "HOLD")
                ai_confidence = decision.get("confidence", 0)
                ai_regime = decision.get("regime", "UNKNOWN")

                ai_result = {
                    "agent_used": True,
                    "agent_action": ai_action,
                    "confidence": ai_confidence,
                    "regime": ai_regime,
                    "tv_action": alert.action,
                    "agreement": ai_action.lower() == alert.action.lower(),
                }

                # AI can override: if agent strongly disagrees, block the trade
                if not ai_result["agreement"] and ai_confidence > 0.7:
                    logger.warning(
                        "AI OVERRIDE: TradingView says %s but agent says %s (confidence=%.2f). BLOCKING.",
                        alert.action, ai_action, ai_confidence,
                    )
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "blocked_by_ai",
                            "tv_action": alert.action,
                            "ai_action": ai_action,
                            "ai_confidence": ai_confidence,
                            "ai_regime": ai_regime,
                            "reason": "AI agent strongly disagrees with TradingView signal",
                        },
                    )

        except Exception as e:
            logger.warning("AI agent processing failed (executing TV signal anyway): %s", e)

        # ── Route to Broker (real execution) ──
        try:
            broker = app.state.broker_router.get_broker(alert.symbol)
            quantity = alert.quantity or 1.0

            order_result = broker.place_order(
                symbol=alert.symbol,
                action=alert.action.lower(),
                quantity=quantity,
                order_type=alert.order_type,
                price=alert.price,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Broker execution failed: {str(e)}")

        app.state.last_alert_time = datetime.now(timezone.utc)
        app.state.total_alerts += 1

        # ── Notifications ──
        if app.state.alert_dispatcher:
            try:
                ai_note = ""
                if ai_result["agent_used"]:
                    ai_note = f"\nAI: {ai_result['agent_action']} (conf={ai_result['confidence']:.0%})"
                app.state.alert_dispatcher.dispatch(
                    title=f"AI Trade: {alert.action.upper()} {alert.symbol}",
                    message=(
                        f"TV Signal: {alert.action} {alert.symbol} @ {alert.price}"
                        f"{ai_note}\nBroker: {broker.name} → {order_result.order_id}"
                    ),
                    level="info",
                )
            except Exception:
                pass

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "alert": {
                    "symbol": alert.symbol,
                    "action": alert.action,
                    "price": alert.price,
                    "quantity": quantity,
                },
                "ai": ai_result,
                "order": order_result.model_dump(),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ─── Status Endpoint ──
    @app.get("/status")
    async def server_status():
        """Get detailed server status including broker connections."""
        brokers = app.state.broker_router.get_all_brokers()
        broker_status = {}
        for name, adapter in brokers.items():
            try:
                info = adapter.get_account_info()
                broker_status[name] = info
            except Exception as e:
                broker_status[name] = {"error": str(e)}

        return {
            "server": "running",
            "uptime_seconds": round(time.time() - app.state.start_time, 2),
            "total_alerts": app.state.total_alerts,
            "brokers": broker_status,
            "rate_limiting": {
                "enabled": config.get("rate_limiting", {}).get("enabled", True),
                "max_per_minute": config.get("rate_limiting", {}).get(
                    "max_requests_per_minute", 60
                ),
            },
        }

    return app


# ─── Application Instance ───
app = create_app()


# ─── Main Entry Point ───
if __name__ == "__main__":
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})

    uvicorn.run(
        "webhook_server:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 5000),
        reload=server_config.get("debug", False),
        workers=server_config.get("workers", 1),
        log_level=config.get("logging", {}).get("level", "info").lower(),
    )
