# ---------------------------- Load Required Packages -------------------------------
install.packages(c("quantmod", "TTR", "caret", "randomForest", "xgboost", "PerformanceAnalytics", "rugarch", "forecast", "ggplot2", "dplyr", "xts"))
install.packages("xgboost")
library(quantmod)
library(TTR)
library(caret)
library(randomForest)
library(xgboost)
library(PerformanceAnalytics)
library(rugarch)
library(forecast)
library(ggplot2)
library(dplyr)
library(xts)
library(fGarch)


library(nnet)              # For simple MLP Neural Network
library(keras)             # Deep Learning with Keras (LSTM, CNN)
library(tensorflow)        # TensorFlow backend for Keras
library(tidyverse)         # For data manipulation


getSymbols("BTC-USD", src = "yahoo", from = "2018-01-01", auto.assign = TRUE)
btc_data <- na.omit(Ad(`BTC-USD`))
colnames(btc_data) <- "Price"

# Compute Log Returns
btc_data$Return <- diff(log(btc_data$Price))
btc_data <- na.omit(btc_data)

# ----------------------------Feature Engineering -------------------------------
btc_data$SMA_10 <- SMA(btc_data$Price, n = 10)
btc_data$SMA_50 <- SMA(btc_data$Price, n = 50)
btc_data$RSI_14 <- RSI(btc_data$Price, n = 14)
btc_data$MACD <- MACD(btc_data$Price)$macd
btc_data$ATR <- ATR(HLC(`BTC-USD`))$atr
btc_data$Volatility <- runSD(btc_data$Return, n = 20)

btc_data <- na.omit(btc_data)


btc_data$Signal <- ifelse(lag(btc_data$Return, 1) > 0, 1, 0)
btc_data <- na.omit(btc_data)

# ----------------------------Train/Test Split -------------------------------
set.seed(42)
train_idx <- createDataPartition(btc_data$Signal, p = 0.8, list = FALSE)
train_data <- btc_data[train_idx, ]
test_data <- btc_data[-train_idx, ]
feature_cols <- setdiff(colnames(train_data), c("Signal", "Price", "Return"))

# ---------------------------- Random Forest Model -------------------------------
rf_model <- randomForest(as.factor(Signal) ~ ., data = train_data[, c(feature_cols, "Signal")], ntree = 100)
rf_pred <- predict(rf_model, newdata = test_data[, feature_cols], type = "class")
rf_pred <- as.numeric(as.character(rf_pred))
saveRDS(rf_model, "bitcoin_rf_model.rds")
# ---------------------------- Boosting (XGBoost) -------------------------------
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, feature_cols]), label = train_data$Signal)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, feature_cols]))
xgb_model <- xgboost(data = train_matrix, objective = "binary:logistic", nrounds = 100, max_depth = 3, eta = 0.1)
xgb_pred <- predict(xgb_model, test_matrix)
xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)
xgb.save(xgb_model, "bitcoin_xgb_model.model")
# ---------------------------- Aparch Model -------------------------------
aparch_model <- garchFit(~ arma(1, 1) + aparch(1, 1), data = train_data$Return, trace = FALSE)

# --------------------- Summary of the fitted APARCH model ---------------------
summary(aparch_model)

# --------------------- Forecasting using APARCH ---------------------
aparch_forecast <- predict(aparch_model, n.ahead = nrow(test_data))

# --------------------- APARCH Mean and Volatility Forecast ---------------------
mean_forecast <- aparch_forecast$meanForecast    # Expected return forecast
vol_forecast <- aparch_forecast$standardDeviation  # Forecasted volatility

# --------------------- APARCH as Directional Signal) ---------------------
aparch_pred <- ifelse(mean_forecast > 0, 1, 0)

# --------------------- APARCH Volatility Adjusted Strategy ---------------------

target_vol <- 0.01

# Volatility-based position sizing 
position_size <- target_vol / vol_forecast

#Limit extreme position sizes (max 5x leverage)
position_size <- pmin(pmax(position_size, 0), 5)

# combine APARCH volatility sizing with ML signals RF:
#
rf_signal <- rf_pred 

# Final APARCH-based strategy using RF direction and APARCH position sizing
test_data$APARCH_Return <- rf_signal * position_size * test_data$Return
test_data$APARCH_Directional_Return <- aparch_pred * test_data$Return



# ---------------------------- LSTM Integration -------------------------------

scaler <- function(x) (x - min(x)) / (max(x) - min(x))
train_scaled <- as.data.frame(lapply(train_data[, feature_cols], scaler))
test_scaled <- as.data.frame(lapply(test_data[, feature_cols], scaler))

X_train <- array(as.matrix(train_scaled), dim = c(nrow(train_scaled), 1, ncol(train_scaled)))
X_test <- array(as.matrix(test_scaled), dim = c(nrow(test_scaled), 1, ncol(test_scaled)))


y_train <- train_data$Signal


k_clear_session()
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(1, length(feature_cols))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

lstm_model %>% compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'accuracy')


lstm_model %>% fit(X_train, y_train, epochs = 20, batch_size = 32, validation_split = 0.2, verbose = 1)


lstm_pred <- ifelse(predict(lstm_model, X_test) > 0.5, 1, 0)
test_data$LSTM_Return <- lstm_pred * test_data$Return  # Add LSTM return

save_model_hdf5(lstm_model, "bitcoin_lstm_model.h5")
# ---------------------------- Buy & Hold Strategy -------------------------------
bh_pred <- rep(1, nrow(test_data))

# ---------------------------- Apply Trading Strategies -------------------------------
test_data$RF_Return <- rf_pred * test_data$Return
test_data$XGB_Return <- xgb_pred * test_data$Return
test_data$BH_Return <- bh_pred * test_data$Return

# ---------------------------- Compute Performance Metrics -------------------------------
perf_metrics <- function(returns, trades = returns) {
  list(
    Sharpe = SharpeRatio.annualized(returns, Rf = 0),
    MaxDD = maxDrawdown(returns),
    WinRate = mean(returns > 0),  
    NumTrades = sum(trades != 0),  
    AvgTradeReturn = mean(returns[trades != 0], na.rm = TRUE),  
    CumulativeReturn = cumprod(1 + returns)[length(returns)],  
    Volatility = sd(returns) * sqrt(252),  
    SortinoRatio = SortinoRatio(returns, Rf = 0), 
    CalmarRatio = Return.annualized(returns) / maxDrawdown(returns)  
  )
}


print("Random Forest Performance:")
print(perf_metrics(test_data$RF_Return))

print("Boosting (XGBoost) Performance:")
print(perf_metrics(test_data$XGB_Return))

print("APARCH Performance with RF Signal:")
print(perf_metrics(test_data$APARCH_Return))

print("APARCH Performance with own Signal:")
print(perf_metrics(test_data$APARCH_Directional_Return))

print("Buy & Hold Performance:")
print(perf_metrics(test_data$BH_Return))

print("LSTM Performance:")
print(perf_metrics(test_data$LSTM_Return))
# ---------------------------- Compare Cumulative Returns -------------------------------
test_data$Cumulative_RF <- cumprod(1 + test_data$RF_Return)
test_data$Cumulative_XGB <- cumprod(1 + test_data$XGB_Return)
test_data$Cumulative_APARCH <- cumprod(1 + test_data$APARCH_Return)
test_data$Cumulative_BH <- cumprod(1 + test_data$BH_Return)
test_data$Cumulative_APARCH_OWN <- cumprod(1 + test_data$APARCH_Directional_Return)
test_data$Cumulative_LSTM <- cumprod(1 + test_data$LSTM_Return)

dates <- index(test_data)
returns_xts <- xts(cbind(test_data$Cumulative_RF, test_data$Cumulative_XGB, 
                         test_data$Cumulative_APARCH, test_data$Cumulative_BH, test_data$Cumulative_APARCH_OWN, test_data$Cumulative_LSTM), 
                   order.by = dates)

colnames(returns_xts) <- c("Random Forest", "Boosting", "APARCH With RF Signals", "Buy & Hold", "APARCH", "LSTM")

# ---------------------------- Visualize Performance -------------------------------
equity_curve <- data.frame(
  Date = index(returns_xts),
  RF = coredata(returns_xts$`Random Forest`),
  XGB = coredata(returns_xts$Boosting),
  APGARCH = coredata(returns_xts$`APARCH With RF Signals`),
  BH = coredata(returns_xts$`Buy & Hold`),
  APARCH_O = coredata(returns_xts$APARCH),
  LSTM = coredata(returns_xts$LSTM)
)

print(head(equity_curve))
print(colnames(equity_curve))

sharpe_rf <- perf_metrics(test_data$RF_Return)$Sharpe
sharpe_xgb <- perf_metrics(test_data$XGB_Return)$Sharpe
sharpe_aparch <- perf_metrics(test_data$APARCH_Directional_Return)$Sharpe
sharpe_bh <- perf_metrics(test_data$BH_Return)$Sharpe
sharpe_aparchRF <- perf_metrics(test_data$APARCH_Return)$Sharpe
sharpe_LSTM <- perf_metrics(test_data$LSTM_Return)$Sharpe

performance_table <- data.frame(
  Model = c("Random Forest", "Boosting", "APARCH with RF", "Buy & Hold", "APARCH", "LSTM"),
  Sharpe = sapply(list(test_data$RF_Return, test_data$XGB_Return, test_data$APARCH_Return, test_data$BH_Return, test_data$APARCH_Directional_Return, test_data$LSTM_Return), 
                  function(x) as.numeric(perf_metrics(returns = x, trades = x)$Sharpe)),
  MaxDD = sapply(list(test_data$RF_Return, test_data$XGB_Return, test_data$APARCH_Return, test_data$BH_Return, test_data$APARCH_Directional_Return, test_data$LSTM_Return), 
                 function(x) as.numeric(perf_metrics(returns = x, trades = x)$MaxDD)),
  WinRate = sapply(list(test_data$RF_Return, test_data$XGB_Return, test_data$APARCH_Return, test_data$BH_Return, test_data$APARCH_Directional_Return, test_data$LSTM_Return), 
                   function(x) perf_metrics(returns = x, trades = x)$WinRate),
  NumTrades = sapply(list(test_data$RF_Return, test_data$XGB_Return, test_data$APARCH_Return, test_data$BH_Return, test_data$APARCH_Directional_Return, test_data$LSTM_Return), 
                     function(x) perf_metrics(returns = x, trades = x)$NumTrades)
)

print(performance_table)
plot_title <- paste0(
  "Trading Strategies vs. Buy & Hold\n",
  "Sharpe Ratios - RF: ", round(sharpe_rf, 2), 
  " | XGB: ", round(sharpe_xgb, 2), 
  " | APARCH with RF: ", round(sharpe_aparchRF, 2), 
  " | BH: ", round(sharpe_bh, 2),
  " | LSTM: ", round(sharpe_LSTM, 2)
)





rf_signals <- data.frame(
  Date = index(test_data),
  Signal = ifelse(rf_pred == 1, "Buy", "Sell"),
  Price = test_data$Cumulative_RF  # Use cumulative return for positioning in the plot
)


rf_signals$Signal_Change <- c(NA, diff(as.numeric(rf_signals$Signal == "Buy")))
rf_trades <- rf_signals[which(rf_signals$Signal_Change != 0), ]

# XGBoost
xgb_signals <- data.frame(
  Date = index(test_data),
  Signal = ifelse(xgb_pred == 1, "Buy", "Sell"),
  Price = test_data$Cumulative_XGB
)
xgb_signals$Signal_Change <- c(NA, diff(as.numeric(xgb_signals$Signal == "Buy")))
xgb_trades <- xgb_signals[which(xgb_signals$Signal_Change != 0), ]

# LSTM
lstm_signals <- data.frame(
  Date = index(test_data),
  Signal = ifelse(lstm_pred == 1, "Buy", "Sell"),
  Price = test_data$Cumulative_LSTM
)
lstm_signals$Signal_Change <- c(NA, diff(as.numeric(lstm_signals$Signal == "Buy")))
lstm_trades <- lstm_signals[which(lstm_signals$Signal_Change != 0), ]


ggplot(equity_curve, aes(x = Date)) +
  geom_line(aes(y = Random.Forest, color = "Random Forest"), size = 1) +
  geom_line(aes(y = Boosting, color = "Boosting"), size = 1) +
  geom_line(aes(y = APARCH, color = "APARCH"), size = 1) +
  geom_line(aes(y = Buy...Hold, color = "Buy & Hold"), size = 1) +
  geom_line(aes(y = APARCH.With.RF.Signals, color = "APARCH With RF Signals"), size = 1) +
  geom_line(aes(y = LSTM, color = "LSTM"), size = 1) +
  labs(title = plot_title, y = "Cumulative Return") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))


