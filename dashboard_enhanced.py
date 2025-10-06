from __future__ import annotations

import logging
import os
import queue
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import requests
from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer, Qt, QMetaObject
from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QSplitter, QHeaderView, QMessageBox, QFileDialog,
    QDoubleSpinBox, QSizePolicy, QCheckBox, QDialog, QFormLayout, QDialogButtonBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import config
from intelligence import TradingBot, ExchangeWrapper, LlamaDecisionEngine, LlamaHybridStrategy

# ─────────────────────────────────────────────────────────────────────────────
# Theme
THEMES = {
    "dark": {
        "bg": "#000000", "fg": "#ffffff", "select_bg": "#1a1a1a", "select_fg": "#ffffff",
        "button_bg": "#0e639c", "button_fg": "#ffffff", "entry_bg": "#0a0a0a", "entry_fg": "#ffffff",
        "text_bg": "#000000", "text_fg": "#d4d4d4", "plot_bg": "#000000", "plot_face": "#0a0a0a",
        "grid_color": "#2a2a2a", "line_colors": [
            "#4CAF50","#2196F3","#FF9800","#E91E63","#9C27B0","#00BCD4","#FFEB3B","#795548","#607D8B","#FFC107"
        ], "total_color": "#F44336"
    },
    "light": {
        "bg": "#f0f0f0", "fg": "#000000", "select_bg": "#e0e0e0", "select_fg": "#000000",
        "button_bg": "#0078d4", "button_fg": "#ffffff", "entry_bg": "#ffffff", "entry_fg": "#000000",
        "text_bg": "#ffffff", "text_fg": "#000000", "plot_bg": "#f5f5f5", "plot_face": "#ffffff",
        "grid_color": "#d0d0d0", "line_colors": [
            "#4CAF50","#2196F3","#FF5722","#9C27B0","#FF9800","#00BCD4","#795548","#607D8B","#E91E63","#FFC107"
        ], "total_color": "#F44336"
    }
}

class QtQueueHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]", level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.log_queue = log_queue
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

# ─────────────────────────────────────────────────────────────────────────────
# Background Workers

# in dashboard_enhanced.py

# in dashboard_enhanced.py
import config

class TradeWorker(QObject):
    done = Signal(list, dict, dict)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._ready = False
        self.bot = None
        self.exchange = None

    def _lazy_init(self):
        if self._ready:
            return
        # Build exchange & bot IN THIS THREAD
        self.exchange = ExchangeWrapper(
            exchange_id=config.EXCHANGE_ID,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            password=getattr(config, "API_PASSWORD", ""),
        )
        # (optional) if your wrapper requires explicit connect():
        if hasattr(self.exchange, "connect"):
            self.exchange.connect()

        # Recreate the same strategy you used in main()
        llama_engine = None
        if getattr(config, "LLAMA_ENABLED", False):
            llama_engine = LlamaDecisionEngine(
                model_name=config.LLAMA_MODEL_NAME,
                base_url=config.LLAMA_API_BASE_URL,
                system_prompt=config.LLAMA_SYSTEM_PROMPT or "",
                request_timeout=config.LLAMA_REQUEST_TIMEOUT,
                max_retries=config.LLAMA_MAX_RETRIES,
                retry_backoff=config.LLAMA_RETRY_BACKOFF,
                options=config.LLAMA_OPTIONS,
            )

        strategy = LlamaHybridStrategy(
            exchange=self.exchange,
            llama_engine=llama_engine,
            coins=config.SENTIMENT_COINS,
            sentiment_prompt=config.LLAMA_SENTIMENT_PROMPT,
            base_currency=config.BASE_CURRENCY,
            buy_threshold=config.SENTIMENT_BUY_THRESHOLD,
            sell_threshold=config.SENTIMENT_SELL_THRESHOLD,
            trade_fraction=config.SENTIMENT_TRADE_FRACTION,
            use_convert=config.LLAMA_SENTIMENT_USE_CONVERT,
            simulate_only=config.LLAMA_SENTIMENT_SIMULATE_ONLY,
            sentiment_weight=getattr(config, "SENTIMENT_WEIGHT", 0.6),
            trend_weight=getattr(config, "TREND_WEIGHT", 0.4),
            require_trend_confirmation=getattr(config, "REQUIRE_TREND_CONFIRMATION", True),
            min_trade_usd=config.MIN_TRADE_USD,
            min_sentiment_delta=config.MIN_SENTIMENT_DELTA,
        )

        self.bot = TradingBot(
            exchange=self.exchange,
            strategy=strategy,
            base_currency=config.BASE_CURRENCY,
            simulate_only=config.LLAMA_SENTIMENT_SIMULATE_ONLY,
        )
        self._ready = True

    @Slot()
    def run_once(self):
        try:
            self._lazy_init()
            trades = self.bot.run_cycle() or []
            balances = self.exchange.get_balances() or {}
            usd_updates = self.exchange.get_balances
            try:
                # Pull latest tickers from Binance
                r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=6)
                if r.ok:
                    price_map = {row["symbol"]: float(row["price"]) for row in r.json()}

                    for coin in balances.keys():
                        sym = f"{coin.upper()}USDT"
                        if sym in price_map:
                            usd_updates[coin.upper()] = float(price_map[sym])
                        # Stablecoins hard-coded to 1.0
                        elif coin.upper() in {"USDT", "USDC", "BUSD"}:
                            usd_updates[coin.upper()] = 1.0
            except Exception:
                # Don’t let a failure here crash the worker
                pass

            self.done.emit(trades, balances, usd_updates)
        except Exception as e:
            self.error.emit(str(e))




class MarketWorker(QObject):
    done = Signal(list)  # list of rows for market table
    error = Signal(str)

    @Slot()
    def refresh(self):
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            tickers = resp.json()
            usdt = [t for t in tickers if t["symbol"].endswith("USDT") and not any(s in t["symbol"] for s in ["USDC","BUSD","TUSD","DAI"])]
            usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
            top = usdt[:20]
            rows = []
            for t in top:
                coin = t["symbol"].replace("USDT", "")
                rows.append([coin, float(t["lastPrice"]), float(t["priceChangePercent"]), float(t["quoteVolume"])])
            self.done.emit(rows)
        except Exception as e:
            self.error.emit(str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Main Window

class TradingApp(QMainWindow):
    def __init__(self, bot: TradingBot, exchange: ExchangeWrapper, log_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self.bot = bot
        self.exchange = exchange
        self.log_queue = log_queue

        self.current_theme = os.getenv("GUI_DEFAULT_THEME", "dark")

        # history for graphing
        self.max_data_points = 100
        self.timestamps = deque(maxlen=self.max_data_points)
        self.balance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_data_points))
        self.total_usd_history = deque(maxlen=self.max_data_points)
        self.show_all_coins = True  # plot all coins lines

        # cache
        self.usd_rates_cache: Dict[str, float] = {}
        self.last_balances: Dict[str, Dict[str, float]] = {}

        # UI refresh
        self.refresh_rate_value = 5

        # build UI
        self.setWindowTitle("CryptoBot2 Dashboard — Threaded")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)
        self._build_ui()
        self._apply_theme()

        # UI timer only repaints from cached data; no network here
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(lambda: self.update_ui(self.last_balances))
        self.update_timer.start(self.refresh_rate_value * 1000)

        # trade cycle driver — triggers worker
        self.bot_running = False
        self.trade_interval_seconds = int(os.getenv("TRADE_INTERVAL", "10"))
        self.trade_timer = QTimer(self)
        self.trade_timer.timeout.connect(self._kick_trade_worker)

        # background threads
        self.trade_thread = QThread(self)
        self.trade_worker = TradeWorker()
        self.trade_worker.moveToThread(self.trade_thread)
        self.trade_worker.done.connect(self._on_trade_done)
        self.trade_worker.error.connect(self._on_trade_error)
        self.trade_thread.start()

        self.mkt_thread = QThread(self)
        self.mkt_worker = MarketWorker()
        self.mkt_worker.moveToThread(self.mkt_thread)
        self.mkt_worker.done.connect(self._on_mkt_rows)
        self.mkt_worker.error.connect(lambda e: logging.getLogger("market").error(e))
        self.mkt_thread.start()

        # refresh market watch every ~30s
        self._market_watch_timer = QTimer(self)
        self._market_watch_timer.timeout.connect(lambda: QMetaObject.invokeMethod(self.mkt_worker, "refresh", Qt.QueuedConnection))
        self._market_watch_timer.start(30_000)
        QMetaObject.invokeMethod(self.mkt_worker, "refresh", Qt.QueuedConnection)

    # ─────────── UI build ───────────
    def _build_ui(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        main = QVBoxLayout(central); main.setContentsMargins(0,0,0,0); main.setSpacing(0)

        # top bar
        top = self._build_control_bar(); main.addWidget(top)

        splitter = QSplitter(Qt.Vertical)

        # upper: graph + balances
        topsec = QWidget(); tl = QHBoxLayout(topsec); tl.setContentsMargins(5,5,5,5)
        hsplit = QSplitter(Qt.Horizontal)
        hsplit.addWidget(self._build_graph_panel())
        hsplit.addWidget(self._build_balances_panel())
        hsplit.setSizes([700,700])
        tl.addWidget(hsplit)
        splitter.addWidget(topsec)

        # lower: log + market watch
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([490,210])
        main.addWidget(splitter)

    def _build_control_bar(self) -> QWidget:
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(10,5,10,5)
        self.start_btn = QPushButton("▶ Start Bot"); self.start_btn.clicked.connect(self.start_bot); lay.addWidget(self.start_btn)
        self.stop_btn  = QPushButton("⏹ Stop Bot"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_bot); lay.addWidget(self.stop_btn)
        self.status_label = QLabel("● Status: Stopped"); f = QFont("Arial", 11); f.setBold(True); self.status_label.setFont(f); lay.addWidget(self.status_label)
        lay.addSpacing(20)
        lay.addWidget(QLabel("Refresh Rate:"))
        self.refresh_spinbox = QSpinBox(); self.refresh_spinbox.setRange(1,60); self.refresh_spinbox.setValue(self.refresh_rate_value); self.refresh_spinbox.valueChanged.connect(self._on_refresh_rate_change); lay.addWidget(self.refresh_spinbox)
        lay.addWidget(QLabel("seconds")); lay.addSpacing(20)
        self.theme_btn = QPushButton("🌙 Dark Mode"); self.theme_btn.clicked.connect(self.toggle_theme); lay.addWidget(self.theme_btn)
        clear_btn = QPushButton("🗑 Clear Graph"); clear_btn.clicked.connect(self.clear_graph_data); lay.addWidget(clear_btn)
        self.usdt_button = QPushButton("Back to USDT"); self.usdt_button.clicked.connect(self.convert_all_to_usdt); lay.addWidget(self.usdt_button)
        self.settings_btn = QPushButton("⚙ Settings"); self.settings_btn.clicked.connect(self.open_settings); lay.addWidget(self.settings_btn)
        lay.addStretch(); return w

    def _build_graph_panel(self) -> QWidget:
        container = QWidget(); l = QVBoxLayout(container); l.setContentsMargins(0,0,0,0)
        self.figure = Figure(figsize=(10,6), dpi=100); self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure); self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l.addWidget(self.canvas)
        legend = QWidget(); self.legend_layout = QHBoxLayout(legend); self.legend_layout.setContentsMargins(5,5,5,5); self.legend_layout.addStretch(); l.addWidget(legend)
        return container

    def _build_balances_panel(self) -> QWidget:
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(5,5,5,5)
        lbl = QLabel("Current Balances"); f = QFont("Arial", 12); f.setBold(True); lbl.setFont(f); l.addWidget(lbl)
        self.balances_table = QTableWidget(); self.balances_table.setColumnCount(3)
        self.balances_table.setHorizontalHeaderLabels(["Currency","Free","USD Value"])
        hdr = self.balances_table.horizontalHeader(); hdr.setSectionResizeMode(QHeaderView.Stretch)
        self.balances_table.setAlternatingRowColors(True); self.balances_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.balances_table.setSelectionBehavior(QTableWidget.SelectRows)
        l.addWidget(self.balances_table); return w

    def _build_log_panel(self) -> QWidget:
        container = QWidget(); l = QHBoxLayout(container); l.setContentsMargins(5,5,5,5)
        # left: log
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(0,0,0,0)
        hdr = QWidget(); hl = QHBoxLayout(hdr); hl.setContentsMargins(0,0,0,0)
        lab = QLabel("Activity Log"); f = QFont("Arial",12); f.setBold(True); lab.setFont(f); hl.addWidget(lab); hl.addStretch()
        save = QPushButton("💾 Save Log"); save.setMaximumWidth(120); save.clicked.connect(self.save_log); hl.addWidget(save)
        ll.addWidget(hdr)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setFont(QFont("Consolas", 10)); ll.addWidget(self.log_text)
        l.addWidget(lw)
        # right: market watch
        mw = QWidget(); ml = QVBoxLayout(mw); ml.setContentsMargins(0,0,0,0)
        mhl = QHBoxLayout(); mlabel = QLabel("Market Watch"); mlabel.setFont(f); mhl.addWidget(mlabel); mhl.addStretch()
        refresh_btn = QPushButton("🔄"); refresh_btn.setMaximumWidth(40); refresh_btn.clicked.connect(lambda: QMetaObject.invokeMethod(self.mkt_worker, "refresh", Qt.QueuedConnection)); mhl.addWidget(refresh_btn)
        ml.addLayout(mhl)
        self.market_table = QTableWidget(); self.market_table.setColumnCount(5)
        self.market_table.setHorizontalHeaderLabels(["Coin","Price","24h %","Volume","Add"])
        hdr2 = self.market_table.horizontalHeader(); hdr2.setSectionResizeMode(0, QHeaderView.ResizeToContents); hdr2.setSectionResizeMode(1, QHeaderView.Stretch); hdr2.setSectionResizeMode(2, QHeaderView.ResizeToContents); hdr2.setSectionResizeMode(3, QHeaderView.Stretch); hdr2.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.market_table.setAlternatingRowColors(True); self.market_table.setEditTriggers(QTableWidget.NoEditTriggers); self.market_table.setSelectionBehavior(QTableWidget.SelectRows)
        ml.addWidget(self.market_table)
        l.addWidget(mw)
        return container

    # ─────────── Threaded trading ───────────
    def _kick_trade_worker(self):
        QMetaObject.invokeMethod(self.trade_worker, "run_once", Qt.QueuedConnection)

    def start_bot(self) -> None:
        if self.bot_running:
            return
        self.bot_running = True
        self.trade_timer.start(max(1, self.trade_interval_seconds) * 1000)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.status_label.setText("● Status: Running")

    def stop_bot(self) -> None:
        if not self.bot_running:
            return
        self.bot_running = False
        self.trade_timer.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.status_label.setText("● Status: Stopped")

    # ─────────── Worker callbacks ───────────
    def _on_trade_done(self, trades: list, balances: dict, usd_updates: dict) -> None:
        if usd_updates:
            self.usd_rates_cache.update(usd_updates)
        if isinstance(balances, dict):
            self.last_balances = balances

        if trades:
            try:
                names = ", ".join(f"{t.get('side', '').upper()} {t.get('symbol', '')}" for t in trades)
                logging.getLogger("gui").info("Executed %d trade(s): %s", len(trades), names)
            except Exception:
                logging.getLogger("gui").info("Executed %d trade(s)", len(trades))

        self.update_ui(self.last_balances)

    def _on_trade_error(self, msg: str) -> None:
        logging.getLogger("gui").error("run_cycle error: %s", msg)

    def _on_mkt_rows(self, rows: List[List[float]]) -> None:
        self.market_table.setRowCount(0)
        for idx, (coin, price, chg, vol) in enumerate(rows):
            self.market_table.insertRow(idx)
            self.market_table.setItem(idx, 0, QTableWidgetItem(str(coin)))
            self.market_table.setItem(idx, 1, QTableWidgetItem(f"${price:,.4f}"))
            item_ch = QTableWidgetItem(f"{chg:+.2f}%"); item_ch.setForeground(QColor("#4CAF50" if chg > 0 else "#F44336"))
            self.market_table.setItem(idx, 2, item_ch)
            vol_str = f"${vol/1e6:.1f}M" if vol >= 1e6 else f"${vol/1e3:.1f}K"
            self.market_table.setItem(idx, 3, QTableWidgetItem(vol_str))
            btn = QPushButton("➕"); btn.setMaximumWidth(40); btn.clicked.connect(lambda _=False, c=coin: self.add_to_opportunistic(c))
            self.market_table.setCellWidget(idx, 4, btn)
            for col in range(4):
                it = self.market_table.item(idx, col)
                if it: it.setTextAlignment(Qt.AlignCenter)

    # ─────────── UI logic ───────────
    def _apply_theme(self) -> None:
        theme = THEMES[self.current_theme]
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(theme["bg"]))
        pal.setColor(QPalette.WindowText, QColor(theme["fg"]))
        pal.setColor(QPalette.Base, QColor(theme["entry_bg"]))
        pal.setColor(QPalette.AlternateBase, QColor(theme["select_bg"]))
        pal.setColor(QPalette.Text, QColor(theme["entry_fg"]))
        pal.setColor(QPalette.Button, QColor(theme["button_bg"]))
        pal.setColor(QPalette.ButtonText, QColor(theme["button_fg"]))
        pal.setColor(QPalette.Highlight, QColor(theme["select_bg"]))
        pal.setColor(QPalette.HighlightedText, QColor(theme["select_fg"]))
        self.setPalette(pal)
        # matplotlib styling
        self.figure.patch.set_facecolor(theme["plot_bg"])
        self.ax.set_facecolor(theme["plot_face"])
        for s in ("bottom","top","left","right"):
            self.ax.spines[s].set_color(theme["fg"])
        self.ax.tick_params(colors=theme["fg"])
        self.ax.xaxis.label.set_color(theme["fg"]) ; self.ax.yaxis.label.set_color(theme["fg"]) ; self.ax.title.set_color(theme["fg"]) ; self.ax.grid(True, alpha=0.3, color=theme["grid_color"])
        self.canvas.draw()

    def toggle_theme(self) -> None:
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.theme_btn.setText("☀️ Light Mode" if self.current_theme == "light" else "🌙 Dark Mode")
        self._apply_theme(); self._update_graph()

    def _on_refresh_rate_change(self, v: int) -> None:
        self.refresh_rate_value = v
        self.update_timer.setInterval(v * 1000)

    def clear_graph_data(self) -> None:
        self.timestamps.clear(); self.balance_history.clear(); self.total_usd_history.clear(); self._update_graph()
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Graph data cleared")

    def save_log(self) -> None:
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save Log File", f"cryptobot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "Text Files (*.txt);;All Files (*)")
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Log saved to {path}")
                QMessageBox.information(self, "Success", f"Log saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log: {e}")

    def convert_all_to_usdt(self) -> None:
        logger = logging.getLogger(__name__)
        if QMessageBox.question(self, "Convert to USDT", "Sell ALL coins to USDT?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            balances = self.exchange.get_balances()
            base = getattr(config, "BASE_CURRENCY", "USDT").upper()
            orders, skipped = [], []
            min_threshold = 10.0
            for coin, info in balances.items():
                if coin.upper() == base:
                    continue
                amount = float(info.get("free", 0.0) or 0.0)
                if amount <= 0: continue
                symbol = f"{coin}/{base}"
                try:
                    t = self.exchange.get_ticker(symbol)
                    px = float(t.get("last") or t.get("close") or 0)
                    if px <= 0: continue
                    notional = amount * px
                    if notional < min_threshold:
                        skipped.append(f"{coin}: ${notional:.2f}"); continue
                    self.exchange.create_order(symbol=symbol, side="sell", amount=amount, order_type="market")
                    orders.append(f"{coin}: {amount:.8f}")
                except Exception as e:
                    logger.error("Failed to sell %s: %s", coin, e)
            msg = []
            if orders: msg += [f"Converted {len(orders)} assets:"] + orders
            if skipped: msg += ["", f"Skipped {len(skipped)} (below $ {min_threshold:.2f}):"] + skipped
            if not orders and not skipped: msg = [f"No assets to convert to {base}"]
            QMessageBox.information(self, "Conversion Complete", "\n".join(msg))
        except Exception as e:
            QMessageBox.critical(self, "Conversion Failed", f"Error: {e}")
            logger.error("ERROR: Conversion failed: %s", e)

    def add_to_opportunistic(self, coin: str):
        try:
            if not hasattr(self.bot.strategy, 'opportunistic_coins'):
                self.bot.strategy.opportunistic_coins = []
            if coin in getattr(self.bot.strategy, 'coins', []) or coin in self.bot.strategy.opportunistic_coins:
                QMessageBox.information(self, "Already Added", f"{coin} is already in your trading list.")
                return
            self.bot.strategy.coins.append(coin)
            self.bot.strategy.opportunistic_coins.append(coin)
            logging.getLogger(__name__).info("Added %s to opportunistic roster", coin)
            QMessageBox.information(self, "Coin Added", f"{coin} added. Will be evaluated next cycle.")
        except Exception as e:
            logging.error("Failed to add %s: %s", coin, e)
            QMessageBox.critical(self, "Error", f"Failed to add coin: {e}")

    # repaint UI from cached balances/rates only (no network)
    def update_ui(self, balances=None) -> None:
        """Periodic UI update: redraw balances table + graph with robust USD pricing."""
        # 0) balances source
        if not isinstance(balances, dict):
            balances = getattr(self, "last_balances", {}) or {}
        else:
            self.last_balances = balances  # keep a copy for next tick

        # 1) Build coin list and fetch robust USD rates (direct + routed + REST fallback)
        try:
            from intelligence import fetch_live_usd_rates
            coins_list = [c.upper() for c in balances.keys() if isinstance(c, str)]
            usd_rates = fetch_live_usd_rates(
                getattr(self.exchange, "client", self.exchange),
                coins_list,
                base_currency=getattr(config, "BASE_CURRENCY", "USDT"),
            ) or {}
        except Exception:
            usd_rates = {}

        # 2) Redraw table
        self.balances_table.setRowCount(0)
        total_usd = 0.0
        row = 0
        coins_present = set()

        for coin, info in balances.items():
            cu = coin.upper()
            free = float(info.get("free", 0.0) or 0.0)
            total_qty = float(info.get("total", free) or 0.0)

            # rate from robust fetcher (fallback 0.0)
            rate = float(usd_rates.get(cu, 0.0) or 0.0)
            usd_value = total_qty * rate

            total_usd += usd_value
            coins_present.add(cu)

            # table row
            self.balances_table.insertRow(row)
            self.balances_table.setItem(row, 0, QTableWidgetItem(cu))
            self.balances_table.setItem(row, 1, QTableWidgetItem(f"{free:.8f}"))
            self.balances_table.setItem(row, 2, QTableWidgetItem(f"${usd_value:,.2f}"))

            # center align
            for c in range(3):
                it = self.balances_table.item(row, c)
                if it:
                    it.setTextAlignment(Qt.AlignCenter)

            row += 1

            # update time-series (by USD value)
            self.balance_history[cu].append(usd_value)

        # 3) TOTAL row
        self.balances_table.insertRow(row)
        total_lbl = QTableWidgetItem("TOTAL")
        total_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        self.balances_table.setItem(row, 0, total_lbl)

        total_val_item = QTableWidgetItem(f"${total_usd:,.2f}")
        total_val_item.setFont(QFont("Arial", 10, QFont.Bold))
        self.balances_table.setItem(row, 2, total_val_item)

        for c in (0, 2):
            it = self.balances_table.item(row, c)
            if it:
                it.setTextAlignment(Qt.AlignCenter)

        # 4) Keep history aligned for coins not present this tick
        for c in list(self.balance_history.keys()):
            if c not in coins_present:
                self.balance_history[c].append(0.0)

        # 5) Update totals + graph + logs
        self.total_usd_history.append(total_usd)
        self.timestamps.append(datetime.now())
        self._update_graph()
        if hasattr(self, "_update_logs"):
            self._update_logs()

    def _update_logs(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if 'BUY' in msg or 'buy' in msg.lower():
                    color = '#4e9cff'
                elif 'SELL' in msg or 'sell' in msg.lower():
                    color = '#ff5252'
                elif 'ERROR' in msg:
                    color = '#ff9800'
                else:
                    color = THEMES[self.current_theme]['text_fg']
                self.log_text.setTextColor(QColor(color))
                ts = datetime.now().strftime("%H:%M:%S")
                if not msg.startswith("["):
                    msg = f"[{ts}] {msg}"
                self.log_text.append(msg)
                text = self.log_text.toPlainText(); lines = text.split('\n')
                if len(lines) > 2000:
                    self.log_text.setPlainText('\n'.join(lines[-2000:]))

    def _update_external_legend(self, legend_items: List[Tuple[str,str]]) -> None:
        while self.legend_layout.count() > 1:
            item = self.legend_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        theme = THEMES[self.current_theme]
        for label, color in legend_items:
            w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(5,2,5,2); h.setSpacing(5)
            box = QLabel(); box.setFixedSize(20,14); box.setStyleSheet(f"background-color:{color}; border-radius:3px; border:1px solid {theme['grid_color']};"); h.addWidget(box)
            txt = QLabel(label); txt.setStyleSheet(f"color:{theme['fg']}; font-size:11px;"); h.addWidget(txt)
            self.legend_layout.insertWidget(self.legend_layout.count()-1, w)

    def _update_graph(self) -> None:
        ax = self.ax; ax.clear()
        if not self.timestamps: return
        theme = THEMES[self.current_theme]
        xs = list(self.timestamps); legend_items = []
        if self.show_all_coins:
            coins_to_plot = list(self.balance_history.keys())
        else:
            latest = {c:(series[-1] if len(series) else 0.0) for c, series in self.balance_history.items()}
            coins_to_plot = [c for c,_ in sorted(latest.items(), key=lambda x:x[1], reverse=True)[:6]]
        for idx, coin in enumerate(coins_to_plot):
            ys = list(self.balance_history[coin])
            if len(xs) == len(ys) and len(xs) >= 2:
                color = theme['line_colors'][idx % len(theme['line_colors'])]
                ax.plot(xs, ys, color=color, linewidth=2)
                legend_items.append((coin, color))
                if ys:
                    ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
                    ax.text(xs[-1], ys[-1], f"  ${ys[-1]:.2f}", color=color, fontsize=9, fontweight='bold', va='center')
        if len(self.total_usd_history) == len(xs) and len(xs) >= 2:
            total = list(self.total_usd_history)
            ax.plot(xs, total, color=theme['total_color'], linewidth=3, linestyle='--')
            legend_items.append(("Total USD", theme['total_color']))
            ax.plot(xs[-1], total[-1], 'o', color=theme['total_color'], markersize=9, markeredgecolor='white', markeredgewidth=2)
            ax.text(xs[-1], total[-1], f"  ${total[-1]:.2f}", color=theme['total_color'], fontsize=10, fontweight='bold', va='center')
        ax.set_title("Portfolio Value Over Time (USD)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=11); ax.set_ylabel("USD ($)", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, alpha=0.25, color=theme['grid_color'])
        self._update_external_legend(legend_items)
        self.canvas.draw_idle()

    # ─────────── Settings dialog (minimal; reuse your existing if desired) ───────────
    def open_settings(self):
        d = QDialog(self); d.setWindowTitle("Bot Settings"); d.setMinimumWidth(420)
        form = QFormLayout(d)
        buy = QDoubleSpinBox(); buy.setRange(0.0, 1.0); buy.setSingleStep(0.01); buy.setValue(getattr(self.bot.strategy,'buy_threshold',0.5)); form.addRow("Buy Threshold:", buy)
        sell = QDoubleSpinBox(); sell.setRange(-1.0, 0.0); sell.setSingleStep(0.01); sell.setValue(getattr(self.bot.strategy,'sell_threshold',-0.4)); form.addRow("Sell Threshold:", sell)
        frac = QDoubleSpinBox(); frac.setRange(0.05, 1.0); frac.setSingleStep(0.05); frac.setValue(getattr(self.bot.strategy,'trade_fraction',0.3)); form.addRow("Trade Fraction:", frac)
        w_s = QDoubleSpinBox(); w_s.setRange(0.0, 1.0); w_s.setSingleStep(0.05); w_s.setValue(getattr(self.bot.strategy,'sentiment_weight',0.6)); form.addRow("Sentiment Weight:", w_s)
        w_t = QDoubleSpinBox(); w_t.setRange(0.0, 1.0); w_t.setSingleStep(0.05); w_t.setValue(getattr(self.bot.strategy,'trend_weight',0.4)); form.addRow("Trend Weight:", w_t)
        req = QCheckBox(); req.setChecked(getattr(self.bot.strategy,'require_trend_confirmation',True)); form.addRow("Require Trend Confirmation:", req)
        cd = QDoubleSpinBox(); cd.setRange(10, 600); cd.setSingleStep(10); cd.setValue(getattr(self.bot.strategy,'cooldown_seconds',180)); form.addRow("Cooldown (s):", cd)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); form.addRow(btns)
        btns.accepted.connect(d.accept); btns.rejected.connect(d.reject)
        if d.exec() == QDialog.Accepted:
            setattr(self.bot.strategy,'buy_threshold',buy.value())
            setattr(self.bot.strategy,'sell_threshold',sell.value())
            setattr(self.bot.strategy,'trade_fraction',frac.value())
            setattr(self.bot.strategy,'sentiment_weight',w_s.value())
            setattr(self.bot.strategy,'trend_weight',w_t.value())
            setattr(self.bot.strategy,'require_trend_confirmation',req.isChecked())
            setattr(self.bot.strategy,'cooldown_seconds',int(cd.value()))
            QMessageBox.information(self, "Updated", "Settings applied.")

# ─────────────────────────────────────────────────────────────────────────────

def run_dashboard(bot: TradingBot, exchange: ExchangeWrapper, log_queue: "queue.Queue[str]") -> None:
    app = QApplication([])
    window = TradingApp(bot, exchange, log_queue)
    window.show()
    app.exec()