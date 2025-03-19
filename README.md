AI-Driven Bitcoin Trading Strategy Using Machine Learning & Time Series Models
Project Overview

This project implements machine learning (ML) and statistical models to develop and evaluate trading strategies for Bitcoin (BTC-USD). The primary goal is to determine whether ML-based models outperform a simple buy-and-hold strategy.

The backtesting framework uses historical price data and incorporates various technical indicators to make trading decisions. Performance is then evaluated using key financial metrics.
🚀 Features & Workflow
1️⃣ Data Collection & Preprocessing

    Historical Bitcoin price data is sourced from Yahoo Finance.
    Data is cleaned and transformed by computing log returns and handling missing values.

2️⃣ Feature Engineering

To improve model performance, the following technical indicators are calculated:

    Simple Moving Averages (SMA 10 & SMA 50) – Trend detection.
    Relative Strength Index (RSI) – Overbought/oversold signals.
    MACD (Moving Average Convergence Divergence) – Momentum-based trend following.
    Average True Range (ATR) – Volatility measure.
    Rolling Standard Deviation (Volatility) – Price fluctuation tracking.

3️⃣ Machine Learning Models

Three different models are used to generate trading signals:

    Random Forest (RF) – A decision-tree-based ensemble method that classifies trends.
    XGBoost (XGB) – A gradient-boosting framework that optimizes trading signals.
    LSTM (Long Short-Term Memory) – A deep learning model for time series forecasting.

4️⃣ Backtesting the Strategy

Each model produces Buy (1) or Sell (0) signals, which are applied to historical BTC price data. Performance is compared to:

    Buy-and-Hold Benchmark – A passive investment strategy.
    Statistical Model (APARCH) – A time-series model for volatility adjustment.

5️⃣ Performance Evaluation

Each strategy is assessed using financial metrics:

    Sharpe Ratio – Risk-adjusted return.
    Max Drawdown – Maximum loss from peak to trough.
    Win Rate – Percentage of profitable trades.
    Annualized Volatility – Risk measure.
    Cumulative Return – Final portfolio growth.



