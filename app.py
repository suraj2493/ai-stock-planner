import pandas as pd
import yfinance as yf


def analyze_stock_agent(ticker_symbol):
  """Fetches live stock data and computes Buy, Stop-Loss, and Target prices using a free data source (yfinance)."""
  print(f"Fetching data for {ticker_symbol.upper()}...")

  # 1. Data Fetching Tool (yfinance)
  stock = yf.Ticker(ticker_symbol)
  df = stock.history(period="6mo")

  if df.empty:
    return f"Error: Could not retrieve data for ticker '{ticker_symbol}'. Please check the symbol."

  # Get Current Market Price (CMP)
  current_price = df["Close"].iloc[-1]

  # 2. Technical & Risk Logic (Calculating ATR for Volatility-based Stop-Loss)
  df["H-L"] = df["High"] - df["Low"]
  df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
  df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
  df["ATR"] = df["TR"].rolling(window=14).mean()

  current_atr = df["ATR"].iloc[-1]

  # Fallback if ATR is somehow NaN
  if pd.isna(current_atr):
    current_atr = current_price * 0.02  # 2% fallback proxy

  # 3. Risk Management Rules
  # Stop Loss: Set 1.5 * ATR below the current market price
  stop_loss = current_price - (1.5 * current_atr)
  risk = current_price - stop_loss

  # Target Price: Calculated using a 1:2 Risk-Reward ratio
  target_price = current_price + (2 * risk)

  # Buy Price: Current Market Price (can be adjusted for a pullback zone)
  buy_price = current_price

  # 4. Structured Output
  recommendation = {
      "Ticker": ticker_symbol.upper(),
      "Current Market Price": round(current_price, 2),
      "Recommended Buy Price": round(buy_price, 2),
      "Stop-Loss Level": round(stop_loss, 2),
      "Target Price": round(target_price, 2),
      "Risk-Reward Ratio": "1:2",
      "Volatility (14-day ATR)": round(current_atr, 2),
  }

  return recommendation


# --- Example Execution ---
if __name__ == "__main__":
  # You can test US stocks ("AAPL", "TSLA"), Indian stocks ("RELIANCE.NS", "TCS.NS"), etc.
  target_ticker = "AAPL"

  result = analyze_stock_agent(target_ticker)

  if isinstance(result, dict):
    print("\n" + "=" * 40)
    print(f" AI STOCK AGENT REPORT: {result['Ticker']}")
    print("=" * 40)
    for key, value in result.items():
      print(f"• {key}: {value}")
    print("=" * 40)
  else:
    print(result)
