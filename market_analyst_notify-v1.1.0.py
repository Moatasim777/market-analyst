"""
Level 1 AI Market Analyst -- Phase 1: Market Intelligence Engine
Version: v1.1.0

WHAT THIS BUILDS ON:
  v1.0.1 fetched candles from Kraken, computed a few indicators, and pushed
  a plain-English report to ntfy.sh. That part is UNCHANGED and still runs.

WHAT'S NEW IN v1.1.0 (Phase 1 of the roadmap):
  A "facts engine" that turns raw candles into a structured JSON snapshot
  per timeframe -- trend, EMA alignment, RSI, MACD, ATR, volume behavior,
  market structure (higher-highs/lower-lows via swing pivots), and
  support/resistance. This JSON is deliberately opinion-free: no "buy",
  no "sell", just facts a trader would look at.

  This JSON is the CONTRACT for Phase 2 -- a future LLM call will receive
  exactly this structure and produce the interpretation, bias, trade
  readiness, confidence, and scenario analysis on top of it. Phase 1 does
  NOT call any LLM. It only produces the facts.

  For now, main() still sends the old-style plain-language ntfy push
  (unchanged behavior), and ALSO prints the new JSON facts to the log so
  you can verify the schema looks right before Phase 2 gets wired in.

SCOPE FOR THIS VERSION:
  - Symbol: BTCUSDT only (deliberately narrow -- prove the schema first).
    SYMBOLS list below is still read from env so you can widen it later
    without code changes, but default is BTC only.
  - Data source: Kraken public API (same as v1.0.1). Binance would need a
    VPS to avoid the cloud-IP block on GitHub Actions -- not worth the
    monthly cost yet since Kraken already works for free. Revisit this
    only if Kraken becomes a real limitation.
  - Gold / other commodities / forex / indices are NOT available on
    Kraken or Binance. When you're ready to add those, this file needs a
    second data source (e.g. Twelve Data or Alpha Vantage have free
    tiers with forex/commodities/indices). That's a separate fetch
    function behind the same JSON contract -- the facts engine below
    doesn't care where candles came from.

No trades are placed. This script only reads public market data.
"""

import os
import json
import requests

KRAKEN_BASE = "https://api.kraken.com/0/public/OHLC"
INTERVAL_MAP = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}
TIMEFRAMES = [("15m", "15M", 1), ("1h", "1H", 2), ("4h", "4H", 3), ("1d", "Daily", 4)]
SYMBOLS = os.environ.get("SYMBOLS", "BTCUSDT").split(",")

SYMBOL_MAP = {"BTCUSDT": "XBTUSDT"}

NTFY_TOPIC = os.environ["NTFY_TOPIC"]


# ---------------------------------------------------------------------------
# Data fetching (unchanged from v1.0.1)
# ---------------------------------------------------------------------------

def fetch_klines(symbol, interval, limit=100):
    kraken_pair = SYMBOL_MAP.get(symbol, symbol)
    kraken_interval = INTERVAL_MAP[interval]
    r = requests.get(
        KRAKEN_BASE,
        params={"pair": kraken_pair, "interval": kraken_interval},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error for {kraken_pair}: {data['error']}")
    pair_keys = [k for k in data["result"].keys() if k != "last"]
    if not pair_keys:
        raise RuntimeError(f"Kraken returned no data for {kraken_pair}")
    raw = data["result"][pair_keys[0]][-limit:]
    # Kraken OHLC row: [time, open, high, low, close, vwap, volume, count]
    return {
        "close": [float(k[4]) for k in raw],
        "high": [float(k[2]) for k in raw],
        "low": [float(k[3]) for k in raw],
        "volume": [float(k[6]) for k in raw],
    }


# ---------------------------------------------------------------------------
# Indicator math
# ---------------------------------------------------------------------------

def ema(values, period):
    k = 2 / (period + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def rsi(values, period=14):
    out = [None] * len(values)
    if len(values) <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains += max(diff, 0)
        losses += max(-diff, 0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = 100 - 100 / (1 + (avg_loss and avg_gain / avg_loss or 100))
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain, loss = max(diff, 0), max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = 100 - 100 / (1 + (avg_loss and avg_gain / avg_loss or 100))
    return out


def macd_hist(values):
    e12, e26 = ema(values, 12), ema(values, 26)
    line = [a - b for a, b in zip(e12, e26)]
    signal = ema(line, 9)
    return [l - s for l, s in zip(line, signal)]


def atr(highs, lows, closes, period=14):
    trs = [None]
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    valid = [t for t in trs if t is not None]
    if len(valid) < period:
        return None
    # Wilder smoothing
    avg = sum(valid[:period]) / period
    for t in valid[period:]:
        avg = (avg * (period - 1) + t) / period
    return avg


def classify_atr(atr_value, price):
    """ATR as % of price -- thresholds are a starting point, not tuned per-asset yet."""
    if atr_value is None or price == 0:
        return "Unknown", None
    pct = (atr_value / price) * 100
    if pct < 0.3:
        return "Low", round(pct, 3)
    if pct < 0.8:
        return "Medium", round(pct, 3)
    return "High", round(pct, 3)


def find_swing_points(highs, lows, window=3):
    """
    Simple fractal-style swing detection: a swing high is a candle whose high
    is the max within `window` candles on both sides; swing low analogous.
    Returns lists of (index, value) for swing highs and swing lows.
    """
    swing_highs, swing_lows = [], []
    n = len(highs)
    for i in range(window, n - window):
        window_highs = highs[i - window:i + window + 1]
        window_lows = lows[i - window:i + window + 1]
        if highs[i] == max(window_highs):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_lows):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def classify_market_structure(highs, lows, window=3):
    """
    Looks at the last two swing highs and last two swing lows to classify
    structure as Higher Highs / Lower Lows / Mixed. This is a starting
    heuristic -- reasonable traders could draw swing points differently,
    which is exactly the kind of ambiguity we hand to the LLM later rather
    than over-engineering in code.
    """
    swing_highs, swing_lows = find_swing_points(highs, lows, window)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "Insufficient Data"

    last_two_highs = [v for _, v in swing_highs[-2:]]
    last_two_lows = [v for _, v in swing_lows[-2:]]

    higher_highs = last_two_highs[1] > last_two_highs[0]
    higher_lows = last_two_lows[1] > last_two_lows[0]

    if higher_highs and higher_lows:
        return "Higher Highs & Higher Lows"
    if not higher_highs and not higher_lows:
        return "Lower Highs & Lower Lows"
    return "Mixed"


def classify_volume(volumes, recent=5, prior=5):
    if len(volumes) < recent + prior:
        return "Unknown"
    recent_avg = sum(volumes[-recent:]) / recent
    prior_avg = sum(volumes[-recent - prior:-recent]) / prior
    if prior_avg == 0:
        return "Unknown"
    ratio = recent_avg / prior_avg
    if ratio > 1.05:
        return "Increasing"
    if ratio < 0.95:
        return "Decreasing"
    return "Flat"


def classify_ema_alignment(price, ema20, ema50):
    if price > ema20 > ema50:
        return "Bullish"
    if price < ema20 < ema50:
        return "Bearish"
    return "Mixed"


def classify_trend(ema_alignment, macd_h, structure):
    """
    Composite trend label from EMA alignment + MACD histogram sign +
    market structure. Simple majority vote across three opinion-free
    signals -- deliberately conservative, defaults to Neutral/Mixed when
    signals disagree rather than forcing a call.
    """
    votes = []
    votes.append(1 if ema_alignment == "Bullish" else -1 if ema_alignment == "Bearish" else 0)
    votes.append(1 if (macd_h is not None and macd_h > 0) else -1 if (macd_h is not None and macd_h < 0) else 0)
    votes.append(1 if structure == "Higher Highs & Higher Lows" else -1 if structure == "Lower Highs & Lower Lows" else 0)
    score = sum(votes)
    if score >= 2:
        return "Bullish"
    if score <= -2:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# Phase 1 core: build the facts JSON for one timeframe / one symbol
# ---------------------------------------------------------------------------

def build_timeframe_facts(symbol, interval):
    k = fetch_klines(symbol, interval, limit=120)
    closes, highs, lows, volumes = k["close"], k["high"], k["low"], k["volume"]
    price = closes[-1]

    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)
    ema20, ema50 = ema20_series[-1], ema50_series[-1]
    ema_alignment = classify_ema_alignment(price, ema20, ema50)

    rsi_val = rsi(closes, 14)[-1]
    macd_h = macd_hist(closes)[-1]
    macd_label = "Bullish" if (macd_h is not None and macd_h > 0) else "Bearish" if (macd_h is not None and macd_h < 0) else "Neutral"

    atr_val = atr(highs, lows, closes, 14)
    atr_label, atr_pct = classify_atr(atr_val, price)

    structure = classify_market_structure(highs, lows)
    volume_label = classify_volume(volumes)

    trend = classify_trend(ema_alignment, macd_h, structure)

    lookback = min(30, len(highs))
    support = min(lows[-lookback:])
    resistance = max(highs[-lookback:])
    distance_to_resistance_pct = round(((resistance - price) / price) * 100, 2) if price else None
    distance_to_support_pct = round(((price - support) / price) * 100, 2) if price else None

    return {
        "trend": trend,
        "ema_alignment": ema_alignment,
        "rsi": round(rsi_val, 1) if rsi_val is not None else None,
        "macd": macd_label,
        "atr": atr_label,
        "atr_pct_of_price": atr_pct,
        "volume": volume_label,
        "market_structure": structure,
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "distance_to_support_pct": distance_to_support_pct,
        "distance_to_resistance_pct": distance_to_resistance_pct,
        "price": round(price, 4),
    }


def build_facts(symbol):
    """
    Produces the Phase 1 -> Phase 2 contract:
    { "symbol": ..., "timeframes": { "15m": {...}, "1h": {...}, "4h": {...}, "1d": {...} } }
    """
    facts = {"symbol": symbol, "timeframes": {}}
    for interval, _label, _weight in TIMEFRAMES:
        facts["timeframes"][interval] = build_timeframe_facts(symbol, interval)
    return facts


# ---------------------------------------------------------------------------
# Existing plain-language report (unchanged behavior from v1.0.1) --
# kept as-is so notifications don't break while Phase 2 isn't wired in yet.
# ---------------------------------------------------------------------------

def analyze_legacy(symbol, interval):
    k = fetch_klines(symbol, interval)
    closes, highs, lows = k["close"], k["high"], k["low"]
    last = closes[-1]
    e20, e50 = ema(closes, 20)[-1], ema(closes, 50)[-1]
    last_rsi = rsi(closes)[-1]
    last_hist = macd_hist(closes)[-1]
    support, resistance = min(lows[-30:]), max(highs[-30:])

    score = 0
    score += 1 if e20 > e50 else -1
    score += 1 if last > e20 else -1
    score += 1 if last_hist > 0 else -1
    trend = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"

    rsi_note = "Overbought" if last_rsi >= 70 else "Oversold" if last_rsi <= 30 else "Neutral"
    return {"trend": trend, "last": last, "rsi": last_rsi, "rsi_note": rsi_note,
            "support": support, "resistance": resistance}


def build_legacy_report(symbol):
    lines = [symbol]
    bull_w = bear_w = total_w = 0
    daily = None
    for interval, label, weight in TIMEFRAMES:
        a = analyze_legacy(symbol, interval)
        if interval == "1d":
            daily = a
        total_w += weight
        if a["trend"] == "Bullish":
            bull_w += weight
        if a["trend"] == "Bearish":
            bear_w += weight
        lines.append(f"{label}: {a['trend']} (RSI {a['rsi']:.0f}{', ' + a['rsi_note'] if a['rsi_note'] != 'Neutral' else ''})")

    alignment = round(max(bull_w, bear_w) / total_w * 100)
    direction = "Bullish" if bull_w > bear_w else "Bearish" if bear_w > bull_w else "Mixed"
    lines.append(f"Alignment: {alignment}% {direction}")
    if daily:
        if direction == "Bullish":
            lines.append(f"Watch: pullback to {daily['support']:.2f} or breakout above {daily['resistance']:.2f}")
        elif direction == "Bearish":
            lines.append(f"Watch: retest of {daily['resistance']:.2f} or breakdown below {daily['support']:.2f}")
        else:
            lines.append("No clear alignment -- stand aside")
    return "\n".join(lines)


def send_ntfy(text):
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    requests.post(
        url,
        data=text.encode("utf-8"),
        headers={"Title": "Level 1 Market Analyst", "Priority": "default", "Tags": "chart_with_upwards_trend"},
        timeout=15,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reports = []
    for sym in SYMBOLS:
        sym = sym.strip()

        # Phase 1 facts engine -- the new part. Logged for now so you can
        # verify the schema; Phase 2 will consume this JSON directly instead
        # of the legacy report below.
        facts = build_facts(sym)
        print(f"--- PHASE 1 FACTS: {sym} ---")
        print(json.dumps(facts, indent=2))

        # Legacy plain-language report -- unchanged, still what gets pushed.
        reports.append(build_legacy_report(sym))

    message = "Report only -- no trades placed.\n\n" + "\n\n".join(reports)
    send_ntfy(message)
    print(message)


if __name__ == "__main__":
    main()
