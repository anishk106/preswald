from preswald import connect, get_df, text, table, slider, select, plotly, view
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

connect()
df = get_df("netflix_stock")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

text("# Netflix Stock Analysis (2002-2025)")
text("""
This interactive dashboard analyzes Netflix stock performance over time, 
providing insights into price trends, returns, volatility, and technical indicators.
Use the controls below to customize your view and analysis parameters.
""")

text("## Stock Data Preview")
table(df.head(10), title="First 10 Rows of Netflix Stock Data")

text("## Analysis Period")
start_date = select(
    "Select Start Date",
    options=list(df['Date'].dt.strftime('%Y-%m-%d').unique())[::30],
    default=df['Date'].min().strftime('%Y-%m-%d')
)

end_date = select(
    "Select End Date",
    options=list(df['Date'].dt.strftime('%Y-%m-%d').unique())[::30],
    default=df['Date'].max().strftime('%Y-%m-%d')
)

filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                 (df['Date'] <= pd.to_datetime(end_date))]

text("## Stock Price Trends")
fig_candlestick = go.Figure(data=[go.Candlestick(
    x=filtered_df['Date'],
    open=filtered_df['Open'],
    high=filtered_df['High'],
    low=filtered_df['Low'],
    close=filtered_df['Close'],
    name='NFLX'
)])
fig_candlestick.update_layout(title='Netflix Stock Price (Candlestick Chart)',
                              xaxis_title='Date',
                              yaxis_title='Price (USD)')
plotly(fig_candlestick)

text("## Trading Volume Analysis")
fig_volume = px.bar(filtered_df, x='Date', y='Volume', 
                   title='Netflix Trading Volume Over Time',
                   color_discrete_sequence=['darkblue'])
plotly(fig_volume)

text("## Technical Analysis")

ma_window = slider("Moving Average Window (Days)", min_val=5, max_val=100, default=20)

filtered_df['SMA'] = filtered_df['Close'].rolling(window=ma_window).mean()
filtered_df['EMA'] = filtered_df['Close'].ewm(span=ma_window, adjust=False).mean()

filtered_df['SMA_20'] = filtered_df['Close'].rolling(window=20).mean()
filtered_df['Upper_Band'] = filtered_df['SMA_20'] + (filtered_df['Close'].rolling(window=20).std() * 2)
filtered_df['Lower_Band'] = filtered_df['SMA_20'] - (filtered_df['Close'].rolling(window=20).std() * 2)

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Close Price'))
fig_ma.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA'], mode='lines', name=f'SMA ({ma_window} days)'))
fig_ma.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['EMA'], mode='lines', name=f'EMA ({ma_window} days)'))
fig_ma.update_layout(title=f'Moving Averages Analysis (Window: {ma_window} days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)')
plotly(fig_ma)

fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Close Price'))
fig_bb.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Upper_Band'], mode='lines', name='Upper Band (+2σ)'))
fig_bb.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA_20'], mode='lines', name='SMA (20 days)'))
fig_bb.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Lower_Band'], mode='lines', name='Lower Band (-2σ)'))
fig_bb.update_layout(title='Bollinger Bands (20-day SMA, 2 standard deviations)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)')
plotly(fig_bb)

text("## Return Analysis")

filtered_df['Daily_Return'] = filtered_df['Close'].pct_change() * 100
filtered_df['Cumulative_Return'] = (1 + filtered_df['Daily_Return'] / 100).cumprod() - 1

fig_returns = px.histogram(filtered_df.dropna(), x='Daily_Return', nbins=50,
                         title='Distribution of Daily Returns (%)',
                         color_discrete_sequence=['green'])
plotly(fig_returns)

fig_cumulative = px.line(filtered_df, x='Date', y='Cumulative_Return',
                        title='Cumulative Return Over Time',
                        labels={'Cumulative_Return': 'Cumulative Return (decimal)'},
                        color_discrete_sequence=['darkgreen'])
fig_cumulative.update_layout(yaxis=dict(tickformat='.2%'))
plotly(fig_cumulative)

text("## Volatility Analysis")

volatility_window = slider("Volatility Calculation Window (Days)", min_val=10, max_val=90, default=30)
filtered_df['Volatility'] = filtered_df['Daily_Return'].rolling(window=volatility_window).std() * np.sqrt(252)

fig_vol = px.line(filtered_df.dropna(), x='Date', y='Volatility',
                title=f'Rolling {volatility_window}-Day Annualized Volatility',
                labels={'Volatility': 'Annualized Volatility'},
                color_discrete_sequence=['red'])
fig_vol.update_layout(yaxis=dict(tickformat='.2%'))
plotly(fig_vol)

text("## Options Pricing Simulation")
text("""
This section simulates the payoff of basic option strategies based on the current 
Netflix stock price and user-defined parameters.
""")

current_price = filtered_df['Close'].iloc[-1]
text(f"Current Netflix Stock Price: ${current_price:.2f}")

strike_price = slider("Strike Price ($)", min_val=float(current_price * 0.5), max_val=float(current_price * 1.5), default=float(current_price))
option_type = select("Option Type", options=["Call", "Put"], default="Call")

price_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)

if option_type == "Call":
    payoffs = [max(price - strike_price, 0) for price in price_range]
    title = f"Call Option Payoff (Strike: ${strike_price:.2f})"
else:
    payoffs = [max(strike_price - price, 0) for price in price_range]
    title = f"Put Option Payoff (Strike: ${strike_price:.2f})"

fig_option = px.line(x=price_range, y=payoffs, 
                   labels={"x": "Stock Price at Expiration", "y": "Option Payoff ($)"},
                   title=title)
fig_option.add_shape(type="line", 
                   x0=current_price, y0=0, 
                   x1=current_price, y1=max(payoffs),
                   line=dict(color="red", width=2, dash="dash"))
fig_option.add_annotation(x=current_price, y=0,
                        text=f"Current Price: ${current_price:.2f}",
                        showarrow=True, arrowhead=1)
plotly(fig_option)

text("## Summary Statistics")

start_date_actual = filtered_df['Date'].min().strftime('%Y-%m-%d')
end_date_actual = filtered_df['Date'].max().strftime('%Y-%m-%d')
total_trading_days = len(filtered_df)
price_change = filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]
percent_change = (price_change / filtered_df['Close'].iloc[0]) * 100
annualized_return = ((1 + percent_change/100) ** (252/total_trading_days) - 1) * 100 if total_trading_days > 0 else 0
max_price = filtered_df['High'].max()
min_price = filtered_df['Low'].min()
avg_volume = filtered_df['Volume'].mean()

stats_df = pd.DataFrame({
    'Metric': ['Analysis Start Date', 'Analysis End Date', 'Total Trading Days', 
              'Starting Price ($)', 'Ending Price ($)', 'Price Change ($)', 
              'Percent Change (%)', 'Annualized Return (%)', 
              'Maximum Price ($)', 'Minimum Price ($)', 'Average Daily Volume'],
    'Value': [start_date_actual, end_date_actual, total_trading_days,
             f"${filtered_df['Close'].iloc[0]:.2f}", f"${filtered_df['Close'].iloc[-1]:.2f}", 
             f"${price_change:.2f}", f"{percent_change:.2f}%", f"{annualized_return:.2f}%",
             f"${max_price:.2f}", f"${min_price:.2f}", f"{avg_volume:,.0f}"]
})

table(stats_df, title="Netflix Stock Performance Summary")

text("### Data Analysis by Preswald")
text("Netflix stock data analysis from 2002-2025")
