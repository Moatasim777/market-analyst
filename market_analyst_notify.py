"""
Level 1 AI Market Analyst -- push notification version (ntfy.sh).

What this does:
  - Pulls 15m/1h/4h/1d candles for a list of symbols from Binance's public API
  - Computes RSI, MACD, EMA20/50, ATR, and recent support/resistance
  - Builds a plain-English report per symbol
  - Pushes it to your phone via ntfy.sh (Telegram is blocked in Pakistan;
    this avoids needing a VPN -- ntfy is a free, open notification service)

Why GitHub Actions:
  You said you don't have a PC yet -- only mobile. GitHub Actions runs this
  on GitHub's servers on a schedule (e.g. every 30 min), so there is no VPS or
  always-on machine to manage. You configure it entirely through the GitHub
  app/website in your phone browser. See SETUP.md for the 10-minute setup.

No trades are placed. This script only reads public market data and sends
a push notification -- it never touches your broker or exchange account.
"""

import os
import requests

BINANCE_BASE = "https://api.binance.com/api/v3/klines"
TIMEFRAMES = [("15m", "15M", 1), ("1h", "1H", 2), ("4h", "4H", 3), ("1d", "Daily", 4)]
SYMBOLS = os.environ.get("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")

# ntfy.sh: pick a hard-to-guess topic name (acts like a private channel).
# Set it as a GitHub Actions secret called NTFY_TOPIC. See SETUP.md.
NTFY_TOPIC = os.environ["NTFY_TOPIC"]


def fetch_klines(symbol, interval, limit=100):
    r = requests.get(BINANCE_BASE, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    return {
        "close": [float(k[4]) for k in raw],
        "high": [float(k[2]) for k in raw],
        "low": [float(k[3]) for k in raw],
    }


def ema(values, period):
    k = 2 / (period + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def rsi(values, period=14):
    out = [None] * len(values)
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


def analyze(symbol, interval):
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


def build_report(symbol):
    lines = [symbol]
    bull_w = bear_w = total_w = 0
    daily = None
    for interval, label, weight in TIMEFRAMES:
        a = analyze(symbol, interval)
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


def main():
    reports = [build_report(sym.strip()) for sym in SYMBOLS]
    message = "Report only -- no trades placed.\n\n" + "\n\n".join(reports)
    send_ntfy(message)
    print(message)


if __name__ == "__main__":
    main()
