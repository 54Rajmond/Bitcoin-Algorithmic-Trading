AI-Driven Bitcoin Trading Strategy Using Machine Learning & Time Series Models
Project Overview

This project leverages machine learning (ML) and statistical models to develop and evaluate algorithmic trading strategies for Bitcoin (BTC-USD). The primary objective is to determine whether ML-based trading strategies can outperform a simple buy-and-hold approach.

Using a comprehensive backtesting framework, historical Bitcoin price data is analyzed to generate trading signals based on various technical indicators. The performance of these strategies is then evaluated using key financial metrics to assess their viability in real-world trading.
Features & Workflow
1. Data Collection & Preprocessing

    Historical Bitcoin price data is sourced from Yahoo Finance.
    The dataset undergoes cleaning and transformation, including:
        Computing log returns for returns-based analysis.
        Handling missing values to ensure data integrity.

2. Feature Engineering

To enhance model performance, several technical indicators are calculated:

    Simple Moving Averages (SMA-10 & SMA-50) – Identifies short- and long-term trends.
    Relative Strength Index (RSI) – Signals overbought or oversold market conditions.
    MACD (Moving Average Convergence Divergence) – Measures momentum for trend following.
    Average True Range (ATR) – Captures market volatility.
    Rolling Standard Deviation (Volatility) – Tracks fluctuations in price movement.

3. Machine Learning Models

Three distinct ML models are employed to generate trading signals:

    Random Forest (RF) – An ensemble learning method based on decision trees for trend classification.
    XGBoost (XGB) – A gradient-boosting framework optimized for predictive accuracy in trading signals.
    LSTM (Long Short-Term Memory) – A deep learning model designed for time-series forecasting and sequential data patterns.

4. Backtesting Strategy

Each model generates Buy (1) or Sell (0) signals, which are applied to historical BTC price data. The effectiveness of these signals is compared against:

    Buy-and-Hold Benchmark – A passive investment strategy.
    Statistical Model (APARCH) – A time-series model that adjusts for volatility.

5. Performance Evaluation

Each strategy is assessed based on critical financial performance metrics, including:

    Sharpe Ratio – Measures risk-adjusted returns.
    Max Drawdown – Evaluates the largest loss from peak to trough.
    Win Rate – Calculates the percentage of profitable trades.
    Annualized Volatility – Quantifies risk exposure.
    Cumulative Return – Measures total portfolio growth over time.

Conclusion

This project aims to determine the effectiveness of AI-driven trading models in generating profitable Bitcoin trading signals. By integrating technical indicators, machine learning algorithms, and statistical methods, we assess whether AI-based strategies can offer a competitive edge over traditional investment approaches.

Further refinements, such as hyperparameter tuning, feature selection optimizations, and real-time data integration, could further enhance model performance and robustness in live trading environments.
Future Enhancements

    Incorporating reinforcement learning techniques for dynamic trading decisions.
    Expanding model testing with alternative cryptocurrencies (e.g., Ethereum, Solana).
    Implementing real-time execution via API connections to live trading platforms.

