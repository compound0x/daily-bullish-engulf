import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io  # Added to fix the Pandas read_html FutureWarning
from IPython.display import display, HTML # Added to render HTML in Colab

def get_sp500_tickers():
    """Automatically grabs the current S&P 500 ticker list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    # FIX: Wrapped response.text in io.StringIO() to resolve the pandas FutureWarning
    table = pd.read_html(io.StringIO(response.text))

    df = table[0]
    return df['Symbol'].tolist()

def scan_and_rank_market(universe_tickers):
    print("Parsing market data and ranking setups... Please wait.\n")
    scored_trades = []

    # Limit to top 50 for speed in Colab. Change to universe_tickers to scan all 500.
    for ticker in universe_tickers[:50]:
        try:
            ticker = ticker.replace('.', '-')
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")

            if len(df) < 200:
                continue

            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()

            current_price = round(df['Close'].iloc[-1], 2)
            open_price = round(df['Open'].iloc[-1], 2)
            sma50 = round(df['SMA50'].iloc[-1], 2)
            sma200 = round(df['SMA200'].iloc[-1], 2)

            dist_from_50 = ((current_price - sma50) / sma50) * 100
            is_green_candle = current_price > open_price

            status = ""
            score = 0
            reason = ""

            if sma50 < sma200:
                status = "4. TRAP (Broken Trend)"
                score = 1
                reason = "The 50 SMA is below the 200 SMA, signaling a macro downtrend. Do not buy pullbacks here."
            elif dist_from_50 < -2:
                status = "4. TRAP (Broken Support)"
                score = 2
                reason = "Price has crashed completely below the 50 SMA support line."
            elif dist_from_50 > 5.5:
                status = "3. TOO EXTENDED (Breakout/Chasing)"
                score = 3
                reason = f"The stock is in a great trend but currently extended {round(dist_from_50,1)}% above the 50 MA. High risk of buying a temporary top."
            elif 2.5 < dist_from_50 <= 5.5:
                status = "2. SLIGHTLY LATE"
                score = 4
                reason = "The 50 MA bounce is working perfectly, but the price has already run up a bit. Entering here requires a wider stop-loss."
            elif 0 <= dist_from_50 <= 2.5 and is_green_candle:
                status = "1. PERFECT MATCH"
                score = 5
                reason = "Textbook setup. Strong macro uptrend, orderly pullback right to the 50 MA, and a decisive green bounce candle today."
            else:
                status = "Unclassified / Consolidating"
                score = 0
                reason = "Price action is neutral or consolidating sideways near moving averages without a clear directional bounce."

            if score >= 3:
                recent_low = df['Low'].iloc[-3:].min()
                cut_loss = round(min(sma50, recent_low) * 0.99, 2)
                recent_high = df['High'].iloc[-20:].max()
                tp1 = round(recent_high * 0.995, 2)
                risk_per_share = current_price - cut_loss
                tp2 = round(current_price + (risk_per_share * 2.0), 2)

                risk_pct = round(((cut_loss - current_price) / current_price) * 100, 1)
                tp1_pct = round(((tp1 - current_price) / current_price) * 100, 1)
                tp2_pct = round(((tp2 - current_price) / current_price) * 100, 1)
                rr_tp1 = round((tp1 - current_price) / risk_per_share, 2) if risk_per_share > 0 else 0
                rr_tp2 = round((tp2 - current_price) / risk_per_share, 2) if risk_per_share > 0 else 0

                scored_trades.append({
                    'ticker': ticker, 'score': score, 'status': status, 'reason': reason,
                    'current_price': current_price, 'sma50': sma50, 'cut_loss': cut_loss,
                    'tp1': tp1, 'tp2': tp2, 'risk_pct': risk_pct, 'tp1_pct': tp1_pct,
                    'tp2_pct': tp2_pct, 'rr_tp1': rr_tp1, 'rr_tp2': rr_tp2
                })
        except:
            continue

    df_results = pd.DataFrame(scored_trades)

    if not df_results.empty:
        df_results = df_results.sort_values(by='score', ascending=False)
        generate_html_report(df_results)
    else:
        print("No qualified setups found in today's scanning parameters.")

def generate_html_report(df_results):
    """Generates an HTML dashboard and renders it in the notebook."""

    html_content = """
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: auto; padding: 20px; background-color: #f4f7f6; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; text-align: center; }
            .rank-card { background: #fff; border: 1px solid #e1e8ed; padding: 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .rank-title { color: #2980b9; margin-top: 0; font-size: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .analysis { background-color: #e8f4f8; padding: 10px; border-left: 4px solid #3498db; margin-bottom: 15px; font-style: italic; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            th { background-color: #2c3e50; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .bold { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Daily Market Analysis & Rankings</h1>
    """

    for idx, row in df_results.iterrows():
        html_content += f"""
        <div class="rank-card">
            <h2 class="rank-title">{row['status']} &rarr; {row['ticker']}</h2>
            <div class="analysis"><strong>The Analysis:</strong> {row['reason']}</div>

            <p><strong>1. Cut Loss (Stop Loss):</strong> ${row['cut_loss']}<br>
            <em>The 50-day SMA is sitting at ${row['sma50']}. Entering at ${row['current_price']} with a stop at ${row['cut_loss']} means risking <strong>{row['risk_pct']}%</strong>.</em></p>

            <p><strong>2. Take Profit Targets:</strong><br>
            &bull; Target 1 (Realistic): <strong>${row['tp1']}</strong> (Projected gain: {row['tp1_pct']}%)<br>
            &bull; Target 2 (Extended): <strong>${row['tp2']}</strong> (Projected gain: {row['tp2_pct']}%)</p>

            <table>
                <tr>
                    <th>Action</th>
                    <th>Price Level</th>
                    <th>Percentage Distance</th>
                    <th>Risk/Reward Ratio</th>
                </tr>
                <tr>
                    <td><strong>Entry Price</strong></td>
                    <td>${row['current_price']}</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td><strong>Cut Loss</strong></td>
                    <td>${row['cut_loss']}</td>
                    <td>{row['risk_pct']}%</td>
                    <td><em>Baseline Risk</em></td>
                </tr>
                <tr>
                    <td><strong>Take Profit 1</strong></td>
                    <td>${row['tp1']}</td>
                    <td>+{row['tp1_pct']}%</td>
                    <td class="bold">{row['rr_tp1']} : 1</td>
                </tr>
                <tr>
                    <td><strong>Take Profit 2</strong></td>
                    <td>${row['tp2']}</td>
                    <td>+{row['tp2_pct']}%</td>
                    <td class="bold">{row['rr_tp2']} : 1</td>
                </tr>
            </table>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Save to an HTML file so you can download it
    with open("daily_market_report.html", "w") as file:
        file.write(html_content)

    # Render it directly in the Colab/Jupyter Notebook
    display(HTML(html_content))
    print("✅ Dashboard generated successfully! You can also download 'daily_market_report.html' from your file explorer.")

# Run the live engine
tickers_to_scan = get_sp500_tickers()
scan_and_rank_market(tickers_to_scan)