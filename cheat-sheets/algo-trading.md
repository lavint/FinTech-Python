# Algorithmic Trading

Utilizing machines to automate the process of buying and selling assets and take human emotion off the equation

* Machines running algorithms can make predictions about ROI, risk, and analyze transactions much faster than a human brain

* Algorithms can be used to predict the best investments based on profit-to-risk ratios, volume, and volatility, or any number of varying attributes

* Using it with portfolio management allows automatic rebalancing of assets and boosts portfolio optimization

* Main benefits:

    1. Can be backtested using historical and real-time data to see if it is a profitable trading strategy

    2. Reduces the possibility of human error in which traders mistime trades based on emotional and psychological factors


## Process

1. Obtain data

2. Make a trading decision with trading signal

3. Evaluate results

<br>


## Syntax highlights

```
# Create dummy business date index
df.index = pd.bdate_range(start='2020-12-12', periods=10)


# Loop through Pandas DataFrame
for index, row in amd_df.iterrows():
    print(row['col1'], index.date())


# Calculate the return on investment
roi = round((total_profit_loss / initial_capital) * 100, 2)  

# Increase Pandas DataFrame display size
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 1000)
```

<br>

## Terminologies

* `Technical Analysis`: Often a short-term trading discipline in which investments are evaluated based on their price action or movement

* `Fundamental analysis`: An investment discipline in which investments are evaluated based on their intrinsic qualities such as financial (income statement, balance sheet, and cash flow statement) or economic data about the underlying company

* `Technical indicator`: A data-driven metric that uses trading data such as closing price and volume to analyze the short or long-term price movements occurring over a specified period

* `Trading signal`: A technical indicator that suggests an opportunity to buy/sell at a point in time and determines trading strategies

* `Short term moving average`: An average price over a short period of time

* `Long term moving average`: An average price over a long period of time

* `Dual moving average crossover points`: A trading signal

    * When the STMA goes above the LTMA, price will rise in the short term, higher than the historical average for that period

    * When the STMA goes below the LTMA, price will drop in the short term, less than the historical average for that period


<br>

## Generate a dual MA crossover trading signal - long position
```
# Get only price data
signals_df = df.loc[:, ["date", "close"]]

# Set the short window and long windows
short_window = 30
long_window = 80

# Set the `date` column as the index
signals_df = signals_df.set_index("date", drop=True)

# Generate the short and long moving averages

signals_df["MA50"] = signals_df["close"].rolling(window=short_window).mean()

signals_df["MA100"] = signals_df["close"].rolling(window=long_window).mean()

signals_df["Signal"] = 0.0

# Generate trading signal with 0 or 1
# 0: when the MA50 is under the MA100, and
# 1: when the MA50 is higher than the MA100

signals_df["Signal"][short_window:] = np.where(
    signals_df["MA50"][short_window:] > signals_df["MA100"][short_window:], 1.0, 0.0
)

# Calculate the points in time at which a position should be taken, 1 or -1
signals_df["Entry/Exit"] = signals_df["Signal"].diff()

```

<br>

## Visualization

```
# Visualize exit position relative to close price

exit = signals_df[signals_df['Entry/Exit'] == -1.0]['close'].hvplot.scatter(
    color='red',
    marker='v',
    size=200,
    legend=False,
    ylabel='Price in $',
    width=1000,
    height=400
)

# Visualize entry position relative to close price

entry = signals_df[signals_df['Entry/Exit'] == 1.0]['close'].hvplot.scatter(
    color='green',
    marker='^',
    size=200,
    legend=False,
    ylabel='Price in $',
    width=1000,
    height=400
)

# Visualize close price for the investment

close = signals_df[['close']].hvplot(
    line_color='lightgray',
    ylabel='Price in $',
    width=1000,
    height=400
)

# Visualize moving averages

moving_avgs = signals_df[['SMA50', 'SMA100']].hvplot(
    ylabel='Price in $',
    width=1000,
    height=400
)

# Overlay plots

entry_exit_plot = close * moving_avgs * entry * exit
entry_exit_plot.opts(xaxis=None)

```

<br>

## Backtesting

* Test the performance of an algorithmic trading strategy using historical stock data

* Help assess the profitability of a trading strategy over time and provide a benchmark for how it may perform going forward

```
# Set initial capital
initial_capital = float(100000)

# Set the share size
# If you are in short position, the share_size = -500
share_size = 500

# Create a position column in which the dual moving average crossover is 1 (MA50 > MA100)
signals_df['Position'] = share_size * signals_df['Signal']


# Find the points in time where a 500 share position is bought or sold
signals_df['Entry/Exit Position'] = signals_df['Position'].diff()


# Get the cumulatively sum
signals_df['Portfolio Holdings'] = signals_df['close'] * signals_df['Entry/Exit Position'].cumsum()


# Get the amount of liquid cash in the portfolio
signals_df['Portfolio Cash'] = initial_capital - (signals_df['close'] * signals_df['Entry/Exit Position']).cumsum()


# Get the total portfolio value
signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']


# Calculate the portfolio daily returns
signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()


# Calculate the cumulative returns
signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1
```




When shorting a stock, it is beneficial to sell the stock at a high price and buy (or cover as they say in the industry) the shares at a low price, resulting in a positive differential. This is why the portfolio cash increases when the algorithm shorts VNQ stock at a high and covers the shares at a low.

