# Machine Learning

Instead of having to configure inputs and manually make changes to an algorithm, machine learning programs automatically adapt to improve outcomes and predictions, as well as accuracy and precision.

The statistical or algorithmic model of the data can be used to make predictions or decisions about new data automatically.

For example, instead of creating some if-else decision structure in order to identify if a transaction is fraudulent, a machine learning algorithm can review all transactions ever made by an account owner, classify and cluster transactions,and then predict whether or not a transaction is fraudulent.

## <u>***Pipeline***</u>
1) Model
2) Fit (Train)
3) Predict

<br>

## <u>***Intelligent algorithms***</u>
* Use pre-existing data to learn and make decisions on how to configure and adapt its behavior for the most accurate and precise prediction
* Are used to fuel machine learning, predictive analytics, and artificial intelligence

1) Supervised learning
    * Potential outcomes need to be known upfront
    * classification, regression

2) Unsupervised learning
    * intelligent algorithm learns on the fly without having seen any type of data before
    * dimensionality reduction and clustering

<br>

## <u>***Predictive Analytics***</u>
* Machine learning is a component of predictive analytics

<br>

## <u>***Artificial Intelligence***</u>
* Machine learning is an application of AI: machines can execute tasks and learn while doing so in order to perform more intelligently
* Instead of programming machines to perform specific tasks, program machines to learn what tasks to complete and how to complete them
* (Deep Learning is also an application of AI)

<br>

## <u>***Time Series***</u>

#### Time series data is decomposed into trend, seasonality, and residual(noise)

## *Trend & Seasonality*
* Decompose time series data
```
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib as mpl
%matplotlib inline


decomposed = seasonal_decompose(df['Sales'], model='multiplicative')

_ = decomposed.plot()

mpl.pyplot.ylim([decomposed.resid.min(), decomposed.resid.max()])

decomposed.observed
decomposed.seasonal

```

* Capture trends and smooth data with moving average
```
# Simple moving average

df['close'].rolling(window=10).mean().plot()

# Exponentially-weighted moving average
# Recent values carry more weight than values from a more distant past
# The shorter the halflife, the more weight the recent observations have  
# Shorter half life means reacting more quickly

df.close.ewm(halflife=10).mean().plot()
```

* Capture trends and smooth data with Hodrick-Prescott filter

    * separates short-term fluctuations from longer-term trends by decomposing a time series into trend and non-trend components

    * minimizes the aggregate values associated with non-trend (periodicity and volatility), thus local fluctuations

```
import statsmodels.api as sm

ts_noise, ts_trend = sm.tsa.filters.hpfilter(df['close'])
```


<br>
<br>


## *Autocorrelation*
#### A measure of how closely current values correlate with past values

## *Partial-Autocorrelation*
#### A measure to identify the number of lags that are significant in explaining the data
#### It reduces components of autocorrelation that are explained by previous lags, and gives heavier weight to lags that have components that are not explained by earlier lags.

* The light blue band is the 95% confidence interval by default
* There is 5% chance that the autocorrelation at a particular lag is found outside the CI band by random chance
* Autocorrelation of lags outside the CI band is significant and not random
* As lag time increases, the CI band widen because as lag is increased, more potential for noise is introduced and the statistical burden of proof is higher

```
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Autocorrelation at lag of 1
df.Temp.autocorr(lag=1)

# Plots
plot_acf(df.Temp, lags=72)
plot_pacf(df.Temp, lags=35, zero=False)

```

<br>
<br>

## *Stationarity*
* A stationary series means the mean and variance are constant over time
* Many time series models assume stationarity
* Use Augmented Dickey-Fuller test to check stationarity

```
# Augmented Dickey-Fuller test
# A p-value (2nd value in the output) < 0.05 means that the series is stationary.

from statsmodels.tsa.stattools import adfuller
adfuller(df.Stationary)
```

* Convert Non-Stationary to Stationary
* NaN and infinity values must be dropped along the way
```
# 1) Apply percent change
df['stationary'] = df['non_stationary'].pct_change()

# 2) Apply diff
df['stationary'] = df['non_stationary'].diff()

# Convert all infinity values to NaNs and drop them
df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan)
df = df.dropna()
```

<br>
<br>

## *Auto Regressive Model*
* Uses past values (& current error) to predict future values
* A linear regression model
* Assumes some degree of auto-correlation
* Uses PACF to estimate the AR order

## *Moving Average Model*
* Uses past errors (& current error) to predict future values
* Not the same as moving/rolling averages which do not take into account the error at each time point
* Uses ACF to estimate the MA order

## *ARMA Model*
* Uses past values and past errors to predict future values
* Assumes stationary series


```
from statsmodels.tsa.arima_model import ARMA

# Create ARMA model using stationary series and the order
# The first order parameter indicates the number of AR lags
# The second order parameter indicates the number of MA lags

model = ARMA(stationary_df.values, order=(1,1))

# Fit the model to the data
results = model.fit()

# Plot the forecasted values for the next 10 days
pd.DataFrame(results.forecast(steps=10)[0]).plot()

# Summarize the model
results.summary()
```


## *ARIMA Model*

* Stationarizes automatically a non-stationary timer series by differencing it

```
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df['Num'], order=(2, 1, 2))
results = model.fit()
results.summary()
pd.DataFrame(results.forecast(steps=3)[0]).plot()
```

* The order refers to AR, diff, and MA components
* Diff refers to the number of times a time series has been differenced to achieve stationarity


<br>

## *Information Criterion*
* AIC (Akaike Information Criterion)
* BIC (Bayesian Information Criterion)
    * estimate the quality of a model
    * favor the simplest model that best fits the data
    * penalize models with a large number of parameters

* Lower AIC and BIC scores are usually better

* A model with a large number of parameters may describe that particular dataset well but may lose its predictive power when used on new data


<br>
<br>

## *GARCH Model (Generalized Autoregressive Conditional Heteroskedasticity)*

* Assumes stationarity
* Forecasts volatility
* Volatility is the change in variance across a time series
* Heteroskedasticity means uneven variance

```
# Create Garch model
from arch import arch_model
model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1)
results = model.fit(disp="off")
results.summary()

# Plot the model estimate of the annualized volatility
fig = results.plot(annualize='D')
```

* Forecasting from GARCH
```
# Construct Volatility Forecasts for the next 3 days

forecast_horizon = 3
last_day = df.index.max().strftime('%Y-%m-%d')
forecasts = results.forecast(start=last_day, horizon=forecast_horizon)

# Annualize the forecast

intermediate = np.sqrt(forecasts.variance.dropna() * 252)


# Each row represents the forecast of volatility for the following days.

final = intermediate.dropna().T
final.plot()
```
