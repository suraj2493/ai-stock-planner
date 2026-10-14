import os
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 1. Page & UI Custom Theme Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Free AI Stock Trade Planner", page_icon="📈", layout="wide"
)

# Dark Terminal Dashboard Aesthetic
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 8px; border: 1px solid #2d3748; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Free AI Stock Trade Planner (Powered by Gemini)")
st.markdown(
    "Enter any stock ticker (e.g., `AAPL`, `RELIANCE.NS`, `TSLA`, `TATAMOTORS.NS`) "
    "to compute technical indicators and receive an AI-driven trade plan completely free."
)


# -----------------------------------------------------------------------------
# 2. Structured Output Schema (Pydantic Model)
# -----------------------------------------------------------------------------
class StockTradePlan(BaseModel):
  ticker: str = Field(description="Stock ticker symbol")
  current_price: float = Field(description="Current market price")
  buy_price: float = Field(description="Recommended entry or buy price")
  stop_loss: float = Field(
      description="Calculated stop loss price based on ATR and support zones"
  )
  target_price: float = Field(
      description="Calculated profit target based on 1:2.5 risk-reward ratio"
  )
  risk_reward_ratio: str = Field(
      description="Risk-to-reward ratio string (e.g., 1:2.5)"
  )
  confidence_score: int = Field(
      description="Agent confidence score ranging from 1 to 10"
  )
  reasoning: str = Field(
      description="Detailed technical rationale and trade execution justification"
  )


# -----------------------------------------------------------------------------
# 3. Deterministic Technical Calculation Engine (Yahoo Finance Math)
# -----------------------------------------------------------------------------
def calculate_technicals(ticker: str) -> dict:
  """Fetches historical price data and computes ATR, support/resistance, and risk parameters."""
  stock = yf.Ticker(ticker)
  df = stock.history(period="3mo")

  if df.empty:
    raise ValueError(
        f"No price data found for ticker '{ticker}'. Please verify ticker spelling."
    )

  # Calculate 14-period Average True Range (ATR)
  df["H-L"] = df["High"] - df["Low"]
  df["H-PC"] = np.abs(df["High"] - df["Close"].shift(1))
  df["L-PC"] = np.abs(df["Low"] - df["Close"].shift(1))
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
  atr = df["TR"].rolling(window=14).mean().iloc[-1]

  current_price = df["Close"].iloc[-1]
  recent_low = df["Low"].tail(20).min()
  recent_high = df["High"].tail(20).max()

  # Set Risk Logic Rules
  buy_price = round(current_price, 2)

  # Stop loss set at 1.5 * ATR below current price or at the 20-day swing low floor
  atr_stop = current_price - (1.5 * atr)
  stop_loss = round(max(atr_stop, recent_low * 0.99), 2)

  # Target calculated based on 1:2.5 Risk-Reward Ratio
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


# -----------------------------------------------------------------------------
