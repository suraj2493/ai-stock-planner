import os
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="AI Stock Trade Planner", page_icon="📈", layout="wide"
)

# Custom Styling (Dark Terminal Theme)
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #2d3748; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📈 AI Stock Trade Planner Agent")
st.markdown(
    "Enter any equity ticker (e.g., `AAPL`, `RELIANCE.NS`, `TSLA`, `TATAMOTORS.NS`) to calculate technical indicators and generate an AI-backed trade plan."
)


# 1. Structured Schema
class StockTradePlan(BaseModel):
  ticker: str = Field(description="Stock ticker symbol")
  current_price: float = Field(description="Current market price")
  buy_price: float = Field(description="Recommended entry or buy price")
  stop_loss: float = Field(
      description="Calculated stop loss price based on ATR/support"
  )
  target_price: float = Field(
      description="Calculated profit target based on risk-reward"
  )
  risk_reward_ratio: str = Field(description="Risk-to-reward ratio (e.g., 1:2.5)")
  confidence_score: int = Field(description="Agent confidence score from 1 to 10")
  reasoning: str = Field(description="Technical justification breakdown")


# 2. Deterministic Technical Math Logic
def calculate_technicals(ticker: str) -> dict:
  stock = yf.Ticker(ticker)
  df = stock.history(period="3mo")

  if df.empty:
    raise ValueError(
        f"No price data found for ticker '{ticker}'. Check symbol spelling."
    )

  df["H-L"] = df["High"] - df["Low"]
  df["H-PC"] = np.abs(df["High"] - df["Close"].shift(1))
  df["L-PC"] = np.abs(df["Low"] - df["Close"].shift(1))
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
  atr = df["TR"].rolling(window=14).mean().iloc[-1]

  current_price = df["Close"].iloc[-1]
  recent_low = df["Low"].tail(20).min()
  recent_high = df["High"].tail(20).max()

  buy_price = round(current_price, 2)
  atr_stop = current_price - (1.5 * atr)
  stop_loss = round(max(atr_stop, recent_low * 0.99), 2)
  risk = buy_price - stop_loss
  target_price = round(buy_price + (risk * 2.5), 2)

  return {
      "ticker": ticker.upper(),
      "current_price": round(current_price, 2),
      "buy_price": buy_price,
      "stop_loss": stop_loss,
      "target_price": target_price,
      "risk_reward_ratio": "1:2.5",
      "atr": round(atr, 2),
      "recent_support": round(recent_low, 2),
      "recent_resistance": round(recent_high, 2),
  }


# 3. Sidebar UI Configuration Inputs
with st.sidebar:
  st.header("⚙️ Configuration")
  ticker_input = st.text_input("Stock Ticker", value="AAPL").upper()
  api_key_input = st.text_input(
      "OpenAI API Key", type="password", placeholder="sk-..."
  )
  run_button = st.button("Generate Trade Plan", type="primary")

# 4. Execution Logic & Web Dashboard Rendering
if run_button:
  if not api_key_input:
    st.error("Please enter your OpenAI API key in the sidebar configuration.")
  else:
    os.environ["OPENAI_API_KEY"] = api_key_input

    try:
      with st.spinner(
          f"Fetching live market data & running agent analysis for"
          f" {ticker_input}..."
      ):
        # Configure PydanticAI Agent (Using output_type instead of result_type)
        stock_agent = Agent(
            "openai:gpt-4o",
            output_type=StockTradePlan,
            system_prompt=(
                "You are an expert quantitative trading assistant. Analyze the"
                " technical math data provided by the tool, verify risk-reward"
                " parameters, assign a realistic confidence score, and craft a"
                " professional trading recommendation breakdown."
            ),
        )

        @stock_agent.tool
        def get_stock_data(ctx: RunContext[None], ticker: str) -> dict:
          return calculate_technicals(ticker)

        prompt = f"Analyze {ticker_input}, fetch its technical indicators, and generate a structured trade plan."
        result = stock_agent.run_sync(prompt)
        plan = result.data

      st.success("Analysis Generated Successfully!")

      # Display Top Metric Cards
      col1, col2, col3, col4 = st.columns(4)
      col1.metric("Current Price", f"${plan.current_price}")
      col2.metric("Buy / Entry Price", f"${plan.buy_price}")
      col3.metric("Stop-Loss", f"${plan.stop_loss}", delta_color="inverse")
      col4.metric("Target Price", f"${plan.target_price}")

      # Detailed Output Section
      st.markdown("---")
      st.subheader("📊 Agent Recommendation & Technical Breakdown")
      col_a, col_b = st.columns(2)
      with col_a:
        st.write(f"**Asset Ticker:** {plan.ticker}")
        st.write(f"**Risk/Reward Structure:** {plan.risk_reward_ratio}")
      with col_b:
        st.metric("Agent Confidence Score", f"{plan.confidence_score} / 10")

      st.info(plan.reasoning)

    except Exception as e:
      st.error(f"An error occurred during execution: {e}")
