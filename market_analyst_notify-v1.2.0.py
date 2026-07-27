"""
Level 1 AI Market Analyst -- Phase 2: LLM Trading Analyst
Version: v1.2.0

WHAT THIS BUILDS ON:
  v1.1.0 added build_facts(symbol) -- a deterministic "facts engine" that
  turns raw candles into structured JSON (trend, EMA alignment, RSI, MACD,
  ATR, volume, market structure, support/resistance) per timeframe. That
  part is UNCHANGED.

WHAT'S NEW IN v1.2.0 (Phase 2 of the roadmap):
  A Claude API call that reads the Phase 1 facts JSON and produces the
  actual analysis -- the part a human trader would be doing in their
  head. The LLM never sees raw candles and never computes indicators
  itself; it only reasons over the facts Python already computed. This
  keeps the assistant's math auditable and its interpretation separate
  from the interpretation itself.

  Three things the system prompt deliberately enforces (per the agreed
  design):
    1. Analyst, not decision-maker -- the LLM interprets facts, it does
       not invent numbers or override the facts engine.
    2. Uncertainty is allowed -- conflicting signals should produce an
       explicit "no clear setup" / "Avoid" readiness, not a forced call.
    3. Objective analysis and interpretation are two separate output
       fields, so the reasoning can always be checked against the facts
       it was reasoning from.
  A scenario analysis (bullish/bearish paths with rough probabilities)
  is also requested, reflecting how a trader actually thinks -- in terms
  of plausible paths, not a single forecast.

  This still does NOT do trade planning (entries/SL/TP) -- that's
  Phase 3, deliberately gated on trade_readiness being "Good" or better.

  Requires an ANTHROPIC_API_KEY secret (in addition to NTFY_TOPIC).

No trades are placed. This script only reads public market data and
calls the Claude API for analysis/interpretation -- it never touches
your broker or exchange account.
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
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = "claude-sonnet-5"


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
# Phase 2: LLM Trading Analyst
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """You are a senior discretionary trader reviewing a pre-computed technical \
snapshot of a market. You are an ANALYST, not a decision-maker: every fact in the input (trend, EMA \
alignment, RSI, MACD, ATR, volume, market structure, support/resistance) was already computed by \
deterministic code and must be treated as ground truth. Do not invent numbers, do not recompute \
indicators, do not contradict the facts you're given -- your job is to explain what they mean and \
whether they add up to something worth acting on.

Think the way an experienced trader actually thinks: in terms of risk/reward, confluence, and \
plausible alternative paths -- not in terms of matching rules to labels.

Hard requirements:
1. Uncertainty is a legitimate, expected answer. When timeframes or indicators disagree, say so \
plainly and let trade_readiness reflect that ("Poor" or "Avoid") instead of forcing a bullish or \
bearish call. "No clear setup, wait" is a correct and valuable output, not a failure to produce one.
2. Keep objective_analysis and interpretation strictly separate. objective_analysis restates the \
facts you were given in plain English with no opinion attached -- someone should be able to check it \
line by line against the input JSON. interpretation is where you reason and give your read.
3. Never state a bias, readiness, or confidence more strongly than the underlying facts support. If \
higher timeframes disagree with lower ones, that disagreement itself is usually the most important \
thing to flag.
4. scenarios must describe at least two plausible paths (typically a bullish and bearish case) with \
rough probabilities that sum to roughly 100. These are not predictions -- they are "if X then \
watch for Y" conditional paths a trader would actually prepare for.
5. Respond with ONLY a single JSON object -- no markdown fences, no preamble, no text outside the \
JSON. It must have exactly this shape:

{
  "objective_analysis": "string, plain-English restatement of the facts across timeframes, no opinion",
  "interpretation": "string, your reasoned read of what it means and why",
  "bias": "Bullish" | "Bearish" | "Neutral",
  "trade_readiness": "Excellent" | "Good" | "Average" | "Poor" | "Avoid",
  "confidence": "High" | "Medium" | "Low",
  "key_risk": "string, the single most important thing that could invalidate this read",
  "scenarios": [
    {"case": "string", "probability": number, "path": "string, what would need to happen"},
    {"case": "string", "probability": number, "path": "string, what would need to happen"}
  ]
}

No trade is ever being placed based on this output -- it is analysis only. Never phrase interpretation \
as an instruction to buy or sell; phrase it as what you would be watching for and why."""


def call_llm_analyst(facts):
    """
    Sends the Phase 1 facts JSON to Claude and returns the parsed analysis
    dict matching ANALYST_SYSTEM_PROMPT's schema. Raises on any failure --
    callers should catch and fall back to the legacy report so a bad API
    call never means no notification at all.
    """
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            # Sonnet 5 has adaptive thinking on by default and thinking
            # tokens count against max_tokens -- keep this generous so a
            # long reasoning pass doesn't truncate the JSON output.
            "max_tokens": 4096,
            "system": ANALYST_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": json.dumps(facts)}
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    text = text.strip()
    if text.startswith("```"):
        # Strip an accidental markdown fence -- the prompt says not to,
        # but models occasionally add one anyway.
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def format_analyst_report(symbol, analysis):
    lines = [symbol, "", "OBJECTIVE ANALYSIS", analysis["objective_analysis"], ""]
    lines.append(
        f"Bias: {analysis['bias']} | Readiness: {analysis['trade_readiness']} | Confidence: {analysis['confidence']}"
    )
    lines.append("")
    lines.append("AI INTERPRETATION")
    lines.append(analysis["interpretation"])
    lines.append("")
    lines.append(f"Key risk: {analysis['key_risk']}")
    scenarios = analysis.get("scenarios") or []
    if scenarios:
        lines.append("")
        lines.append("SCENARIOS")
        for s in scenarios:
            prob = s.get("probability")
            prob_str = f"{prob}%" if prob is not None else "?"
            lines.append(f"- {s.get('case', '?')} ({prob_str}): {s.get('path', '')}")
    return "\n".join(lines)


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

        facts = build_facts(sym)
        print(f"--- PHASE 1 FACTS: {sym} ---")
        print(json.dumps(facts, indent=2))

        try:
            analysis = call_llm_analyst(facts)
            print(f"--- PHASE 2 ANALYSIS: {sym} ---")
            print(json.dumps(analysis, indent=2))
            reports.append(format_analyst_report(sym, analysis))
        except Exception as e:
            # Never let an LLM/API hiccup mean silence -- fall back to the
            # deterministic legacy report so a notification still goes out.
            print(f"--- PHASE 2 FAILED for {sym}, falling back to legacy report: {e} ---")
            reports.append(build_legacy_report(sym))

    message = "Report only -- no trades placed.\n\n" + "\n\n".join(reports)
    send_ntfy(message)
    print(message)


if __name__ == "__main__":
    main()
