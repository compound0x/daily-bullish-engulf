import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


OUTPUT_DIR = Path("strict-bullish-engulf-report")
OUTPUT_FILE = OUTPUT_DIR / "index.html"

MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000
MAX_ATR_PCT = 0.08
MAX_GAP_PCT = 0.03
MIN_RR = 1.5
MAX_RISK_PCT = 0.05


def get_sp500_tickers():
    """Return the current S&P 500 ticker universe."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    return tables[0]["Symbol"].tolist()


def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()
    df["VOL20"] = volume.rolling(20).mean()

    return df


def get_market_regime():
    """Classify the broad market using SPY daily trend."""
    spy = yf.download("SPY", period="1y", interval="1d", auto_adjust=False, progress=False)
    if spy.empty:
        return "UNKNOWN", 0

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = add_indicators(spy.dropna(subset=["Close"]))
    if len(spy) < 200:
        return "UNKNOWN", 0

    latest = spy.iloc[-1]
    slope_20 = spy["SMA50"].iloc[-1] - spy["SMA50"].iloc[-21]

    conditions = [
        latest["Close"] > latest["SMA50"],
        latest["SMA50"] > latest["SMA200"],
        slope_20 > 0,
    ]
    passed = int(sum(conditions))

    if passed == 3:
        return "RISK-ON", 3
    if passed == 2:
        return "NEUTRAL", 2
    return "RISK-OFF", passed


def relative_strength_vs_spy(stock_df, spy_df):
    if len(stock_df) < 21 or len(spy_df) < 21:
        return np.nan
    stock_return = stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-21] - 1
    spy_return = spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-21] - 1
    return stock_return - spy_return


def get_spy_data():
    spy = yf.download("SPY", period="1y", interval="1d", auto_adjust=False, progress=False)
    if spy.empty:
        return spy
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return add_indicators(spy.dropna(subset=["Close"]))


def score_setup(row, market_score):
    score = 0

    trend_spread = row["trend_spread"]
    if trend_spread >= 0.05:
        score += 5
    elif trend_spread >= 0.02:
        score += 3

    if row["sma50_slope"] > 0:
        score += 5
    if row["sma200_slope"] >= 0:
        score += 5
    if row["close_above_sma200"]:
        score += 5

    location_atr = row["location_atr"]
    if location_atr <= 0.5:
        score += 10
    elif location_atr <= 1.0:
        score += 6

    if 0.02 <= row["pullback_pct"] <= 0.06:
        score += 5
    elif row["pullback_pct"] <= 0.10:
        score += 3

    if row["body_atr"] >= 0.75:
        score += 5
    elif row["body_atr"] >= 0.50:
        score += 3

    if row["close_location"] >= 0.90:
        score += 5
    elif row["close_location"] >= 0.75:
        score += 3

    if row["upper_wick_body"] <= 0.35:
        score += 5
    elif row["upper_wick_body"] <= 0.75:
        score += 3

    if row["volume_ratio"] >= 1.5:
        score += 10
    elif row["volume_ratio"] >= 1.2:
        score += 6
    else:
        score += 3

    if row["relative_strength"] >= 0.05:
        score += 10
    elif row["relative_strength"] >= 0.02:
        score += 6
    elif row["relative_strength"] >= 0:
        score += 3

    if market_score == 3:
        score += 10
    elif market_score == 2:
        score += 5

    if row["rr"] >= 3:
        score += 10
    elif row["rr"] >= 2:
        score += 6
    else:
        score += 3

    return int(score)


def scan_stock(ticker, spy_df):
    ticker = ticker.replace(".", "-")
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y", interval="1d", auto_adjust=False)
    if df.empty or len(df) < 220:
        return None

    df = add_indicators(df).dropna(subset=["SMA20", "SMA50", "SMA200", "ATR14", "VOL20"])
    if len(df) < 30:
        return None

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    close = float(cur["Close"])
    open_ = float(cur["Open"])
    high = float(cur["High"])
    low = float(cur["Low"])
    atr = float(cur["ATR14"])
    sma20 = float(cur["SMA20"])
    sma50 = float(cur["SMA50"])
    sma200 = float(cur["SMA200"])

    dollar_volume = close * float(cur["Volume"])
    if close <= MIN_PRICE or dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    atr_pct = atr / close
    if atr_pct > MAX_ATR_PCT:
        return None

    gap_pct = abs(open_ / float(prev["Close"]) - 1)
    if gap_pct > MAX_GAP_PCT:
        return None

    sma50_slope = sma50 - float(df["SMA50"].iloc[-21])
    sma200_slope = sma200 - float(df["SMA200"].iloc[-21])
    trend_spread = (sma50 - sma200) / sma200

    if not (close > sma200 and sma50 > sma200 and sma50_slope > 0 and sma200_slope >= 0):
        return None
    if trend_spread < 0.02:
        return None

    recent_high_10 = float(df["High"].iloc[-11:-1].max())
    pullback_pct = (recent_high_10 - float(prev["Close"])) / recent_high_10
    if not (0.02 <= pullback_pct <= 0.10):
        return None

    dist_20 = abs(close - float(cur["EMA20"])) / atr
    dist_50 = abs(close - sma50) / atr
    location_atr = min(dist_20, dist_50)
    if location_atr > 1.0:
        return None

    extension_atr = (close - float(cur["EMA20"])) / atr
    if extension_atr > 2.0:
        return None

    prev_bearish = float(prev["Close"]) < float(prev["Open"])
    cur_bullish = close > open_
    body_engulfs = open_ <= float(prev["Close"]) and close >= float(prev["Open"])
    closes_above_prev_high = close > float(prev["High"])
    bullish_engulfing = prev_bearish and cur_bullish and body_engulfs and closes_above_prev_high
    if not bullish_engulfing:
        return None

    body = close - open_
    if body <= 0:
        return None
    body_atr = body / atr
    if body_atr < 0.50:
        return None

    candle_range = max(high - low, 1e-9)
    close_location = (close - low) / candle_range
    if close_location < 0.75:
        return None

    upper_wick = max(0.0, high - close)
    upper_wick_body = upper_wick / body
    if upper_wick_body > 0.75:
        return None

    volume_ratio = float(cur["Volume"]) / float(cur["VOL20"])
    if volume_ratio < 1.0:
        return None

    relative_strength = relative_strength_vs_spy(df, spy_df)
    if not np.isfinite(relative_strength) or relative_strength < 0:
        return None

    recent_swing_low = float(df["Low"].iloc[-5:].min())
    stop = min(recent_swing_low - 0.25 * atr, sma50 - 0.25 * atr)
    risk = close - stop
    if risk <= 0:
        return None

    risk_pct = risk / close
    if risk_pct > MAX_RISK_PCT:
        return None

    resistance_candidates = [
        float(df["High"].iloc[-20:].max()),
        float(df["High"].iloc[-50:].max()),
    ]
    resistances = [r for r in resistance_candidates if r > close]
    if not resistances:
        return None
    resistance = min(resistances)

    reward = resistance - close
    rr = reward / risk
    if rr < MIN_RR:
        return None

    row = {
        "ticker": ticker,
        "current_price": close,
        "stop": stop,
        "resistance": resistance,
        "risk_pct": risk_pct,
        "rr": rr,
        "atr_pct": atr_pct,
        "trend_spread": trend_spread,
        "sma50_slope": sma50_slope,
        "sma200_slope": sma200_slope,
        "close_above_sma200": close > sma200,
        "pullback_pct": pullback_pct,
        "location_atr": location_atr,
        "extension_atr": extension_atr,
        "body_atr": body_atr,
        "close_location": close_location,
        "upper_wick_body": upper_wick_body,
        "volume_ratio": volume_ratio,
        "relative_strength": relative_strength,
    }
    return row


def fmt_pct(value):
    return f"{value * 100:.1f}%"


def generate_html(results, market_regime, market_score):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not results:
        rows_html = "<div class='empty'>No setups passed all strict filters today.</div>"
    else:
        cards = []
        for idx, row in enumerate(results, start=1):
            grade = "A+" if row["score"] >= 90 else "A" if row["score"] >= 80 else "B"
            target = row["resistance"]
            cards.append(
                f"""
                <section class='card'>
                  <div class='rank'>#{idx}</div>
                  <div class='header'>
                    <div><h2>{row['ticker']}</h2><span class='grade'>{grade}</span></div>
                    <div class='score'>{row['score']}/100</div>
                  </div>
                  <p class='summary'>Strong daily bullish-engulfing setup after a controlled pullback, with trend, volume, relative-strength and reward/risk confirmation.</p>
                  <div class='grid'>
                    <div><b>Entry</b><span>${row['current_price']:.2f}</span></div>
                    <div><b>Stop</b><span>${row['stop']:.2f}</span></div>
                    <div><b>Resistance</b><span>${target:.2f}</span></div>
                    <div><b>Risk</b><span>{fmt_pct(row['risk_pct'])}</span></div>
                    <div><b>R:R</b><span>{row['rr']:.2f}:1</span></div>
                    <div><b>Volume</b><span>{row['volume_ratio']:.2f}x</span></div>
                    <div><b>Rel. Strength</b><span>{fmt_pct(row['relative_strength'])}</span></div>
                    <div><b>ATR</b><span>{fmt_pct(row['atr_pct'])}</span></div>
                  </div>
                  <div class='details'>
                    <span>Trend spread: {fmt_pct(row['trend_spread'])}</span>
                    <span>Pullback: {fmt_pct(row['pullback_pct'])}</span>
                    <span>Support distance: {row['location_atr']:.2f} ATR</span>
                    <span>Engulf body: {row['body_atr']:.2f} ATR</span>
                    <span>Close location: {row['close_location']*100:.0f}%</span>
                  </div>
                </section>
                """
            )
        rows_html = "\n".join(cards)

    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Strict Bullish Engulfing Scanner</title>
<style>
body{{font-family:Arial,sans-serif;background:#f5f7fa;color:#1f2937;max-width:1000px;margin:auto;padding:24px}}
h1{{margin-bottom:6px}} .sub{{color:#6b7280;margin-top:0}}
.regime{{background:#fff;padding:16px;border-radius:12px;margin:18px 0;border:1px solid #e5e7eb}}
.card{{position:relative;background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 3px 10px rgba(0,0,0,.04)}}
.rank{{position:absolute;right:18px;top:18px;color:#6b7280;font-weight:700}}
.header{{display:flex;justify-content:space-between;align-items:center;padding-right:45px}} h2{{margin:0 0 6px}} .grade{{font-weight:700}} .score{{font-size:28px;font-weight:800}}
.summary{{color:#4b5563}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}
.grid div{{background:#f8fafc;padding:12px;border-radius:8px}} .grid b{{display:block;font-size:12px;color:#6b7280;margin-bottom:5px}} .grid span{{font-weight:700}}
.details{{display:flex;flex-wrap:wrap;gap:14px;margin-top:15px;font-size:13px;color:#6b7280}} .empty{{background:#fff;padding:24px;border-radius:12px;border:1px solid #e5e7eb}}
@media(max-width:700px){{.grid{{grid-template-columns:repeat(2,1fr)}} body{{padding:14px}}}}
</style>
</head>
<body>
<h1>Strict Bullish Engulfing Scanner</h1>
<p class='sub'>Daily S&P 500 scanner with hard filters before ranking.</p>
<div class='regime'><b>Market regime:</b> {market_regime} &nbsp; | &nbsp; <b>SPY conditions:</b> {market_score}/3 &nbsp; | &nbsp; <b>Qualified setups:</b> {len(results)}</div>
{rows_html}
</body>
</html>"""
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Saved {OUTPUT_FILE}")


def main():
    print("Running strict bullish engulfing scanner...")
    market_regime, market_score = get_market_regime()
    if market_regime == "UNKNOWN":
        print("Could not determine SPY market regime; aborting.")
        return
    if market_regime == "RISK-OFF":
        print("Market regime is RISK-OFF. No long setups will be published.")
        generate_html([], market_regime, market_score)
        return

    spy_df = get_spy_data()
    tickers = get_sp500_tickers()
    results = []

    for ticker in tickers:
        try:
            row = scan_stock(ticker, spy_df)
            if row is not None:
                row["score"] = score_setup(row, market_score)
                if row["score"] >= 70:
                    results.append(row)
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")

    results.sort(key=lambda x: (x["score"], x["rr"], x["volume_ratio"]), reverse=True)
    generate_html(results, market_regime, market_score)


if __name__ == "__main__":
    main()
