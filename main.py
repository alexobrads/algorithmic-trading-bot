import mlflow
import random

mlflow.set_tracking_uri("http://localhost:8080")

# Simulate different trading strategies
def simulate_trading_strategy(strategy_name, trading_pair, timeframe, risk_tolerance):
    # Simulate results for different strategies
    profit_loss = random.uniform(-1000, 5000)  # Simulated P/L
    max_drawdown = random.uniform(0.01, 0.3)   # Simulated max drawdown
    sharpe_ratio = random.uniform(-1, 3)       # Simulated Sharpe ratio
    num_trades = random.randint(5, 100)        # Number of trades
    win_rate = random.uniform(0.3, 0.8)        # Simulated win rate

    return profit_loss, max_drawdown, sharpe_ratio, num_trades, win_rate

# Start a new MLflow run for each strategy
strategy_name = "Moving Average Crossover"
trading_pair = "BTC/USD"
timeframe = "1h"
risk_tolerance = 0.02  # 2% risk per trade

with mlflow.start_run():
    # Log strategy parameters
    mlflow.log_param("strategy_name", strategy_name)
    mlflow.log_param("trading_pair", trading_pair)
    mlflow.log_param("timeframe", timeframe)
    mlflow.log_param("risk_tolerance", risk_tolerance)

    # Simulate the strategy and get performance metrics
    profit_loss, max_drawdown, sharpe_ratio, num_trades, win_rate = simulate_trading_strategy(
        strategy_name, trading_pair, timeframe, risk_tolerance
    )

    # Log metrics
    mlflow.log_metric("profit_loss", profit_loss)
    mlflow.log_metric("max_drawdown", max_drawdown)
    mlflow.log_metric("sharpe_ratio", sharpe_ratio)
    mlflow.log_metric("num_trades", num_trades)
    mlflow.log_metric("win_rate", win_rate)
