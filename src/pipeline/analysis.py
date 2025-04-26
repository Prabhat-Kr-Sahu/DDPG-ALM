import pandas as pd

# Load data
import os
path=os.path.join('artifacts','df_daily_return.csv')
data = pd.read_csv(path)  # Replace with the actual file or data you pasted
data['daily_return'] = data['daily_return'].astype(float)

# Initial conditions
initial_value = 1000
data['portfolio_value'] = initial_value * (1 + data['daily_return']).cumprod()

# Statistics
final_value = data['portfolio_value'].iloc[-1]
total_return = (final_value - initial_value) / initial_value * 100
average_daily_return = data['daily_return'].mean() * 100
std_daily_return = data['daily_return'].std() * 100

# Max Drawdown
data['running_max'] = data['portfolio_value'].cummax()
data['drawdown'] = (data['portfolio_value'] - data['running_max']) / data['running_max']
max_drawdown = data['drawdown'].min() * 100

# Annualized Return and Volatility
annualized_return = ((final_value / initial_value) ** (252 / len(data)) - 1) * 100
annualized_volatility = std_daily_return * (252 ** 0.5)

# Output
stats = {
    'Final Portfolio Value ($)': final_value,
    'Total Return (%)': total_return,
    'Average Daily Return (%)': average_daily_return,
    'Daily Volatility (%)': std_daily_return,
    'Maximum Drawdown (%)': max_drawdown,
    'Annualized Return (%)': annualized_return,
    'Annualized Volatility (%)': annualized_volatility
}

for key, value in stats.items():
    print(f"{key}: {value:.2f}")
