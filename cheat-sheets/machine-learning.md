# Machine Learning

Instead of having to configure inputs and manually make changes to an algorithm, machine learning programs automatically adapt to improve outcomes and predictions, as well as accuracy and precision.

The statistical or algorithmic model of the data can be used to make predictions or decisions about new data automatically.

For example, instead of creating some if-else decision structure in order to identify if a transaction is fraudulent, a machine learning algorithm can review all transactions ever made by an account owner, classify and cluster transactions,and then predict whether or not a transaction is fraudulent.

## <u>***Pipeline***</u>
1) Preprocess/Clean data
2) Train/Fit
3) Validate
4) Predict

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

## *Autoregression Model*
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

* AIC and BIC reward models for fitting data accurately, but punish them for having a large number of parameters


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

<br>

## <u>***Linear Regression***</u>

* Identifies the line that best predict `y` based on the value of `X`
* Models data with a linear trend (1 straight line)
* Residual: the difference between the predicted `y` and actual `y`
* Finds the best fit line by minimizing the sum of square value of residuals/errors
* Performs predictive analysis
```
y = mx + b

# y is the dependent variable
# x is the independent variable
```

Scikit learn takes `y` in pandas series type and `X` in shape (n, 1)

```
# Reformat x data points to use Scikit learn
X = df.independent_column.values.reshape(-1, 1)
```


Create and train a linear model
```
model = LinearRegression()  # a straight line
model.fit(X,y)                  

print(model.coef_)        # slope
print(model.intercept_)   # y-intercept
```

Predict `y` based on `X` 
* Given an x that is not in the dataset, model will predict the corresponding y
```
predicted_y_values = model.predict(X)
```

Check accuracy

1) Mean Squared Error
    * The variance of the errors in the dataset
    * Lower the error, higher the accuracy

2) Root Mean Squared Error
    * The standard deviation of the errors in the dataset
    * Lower the error, higher the accuracy

3) R2
    * The square of the correlation coefficient
    * Describes the extent to which a change in one variable is associated with the change in the other variable


```
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# R2 values
score = model.score(X_binary_encoded, y)

# Another way to do R2 values
r2 = r2_score(y, predictions)

# MSE
mse = mean_squared_error(y, predictions)

# Ideally, the RMSE will not exceed the standard deviation
# RMSE exceeds SD indicates that the model is not very helpful
# On average, there are wider swings in errors than measured values

rmse = np.sqrt(mse)
np.std(y)
```

Plot the result
```
plt.scatter(X, y)
plt.plot(X, predicted_y_values, color='red')
```


<br>

## <u>***Time Series Linear Regression***</u>

Use datetime attributes to create a new column
```
X['Week_of_Year'] = X.index.weekofyear
```

Convert categorical data to numeric data
```
# creates a column for each week of the year
X_binary_encoded = pd.get_dummies(X, columns=['Week_of_Year'])


# Each week is a separate variable in the equation
# It is a multiple regression
```



<br>



## <u>***Underfitting***</u>
* Occurs when a model is too generalized to identify underlying pattern
* The bias is high, meaning that the model is not sophisticated enough to capture the general pattern of the data
* The model is too simple


<br>

### *When choosing a model, it is important to keep in mind the balance between bias and variance*
### *When two models perform similarly, choose the simpler one*


<br>

## <u>***Overfitting***</u>
* Occurs when a model is too specific to a particular data set
* Model memorizes the random patterns of the training data too well
* It memorizes the quirks of a dataset without identifying the underlying patterns
* It learns the noise found in the training data, rather than just the signal
* The variance (prediction error on new data) is high, meaning that the model will not be generalizable to other contexts
* The model is too complex

Cause:
* *Excessive number of variables* : resulting in a rigid model that does not generalize well


Solution:
* Train-test-split



## <u>***Train-Test-Split***</u>

* Split a dataset into 80% training set and 20% testing set
* Train set: learns the relevant patterns and minimize errors
* Test set: evaluate the model's performance on unseen data

```
X_train = train["Lagged_Return"].to_frame()
X_test = test["Lagged_Return"].to_frame()
y_train = train["Return"]
y_test = test["Return"]


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


Results = y_test.to_frame()
Results["Predicted Return"] = predictions
```



<br>

## <u>***Model Performance***</u>

Out-of-sample (Test) performance
```
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(
    Results["Return"],
    Results["Predicted Return"])
rmse = np.sqrt(mse)
```



In-sample (Train) performance
```
from sklearn.metrics import mean_squared_error, r2_score
in_sample_results = y_train.to_frame()
in_sample_results["In-sample Predictions"] = model.predict(X_train)
in_sample_mse = mean_squared_error(
    in_sample_results["Return"],
    in_sample_results["In-sample Predictions"])
in_sample_rmse = np.sqrt(in_sample_mse)
```



<br>

## <u>***Rolling windows with linear regression***</u>



* The start and end of training period are defined with weeks[i] and weeks[training_window+i], respectively
* The start and end of testing period are defined as weeks[training_window+i+1]. This is the week after (and only the week after) the training window ends

```
# Split the index into weekly periods
weeks = df.index.to_period("w").unique()

# Declare variables
training_window = 18
timeframe = len(weeks) - training_window - 1

# Construct empty dataframe
all_predictions = pd.DataFrame(columns=["Out-of-Sample Predictions"])
all_actuals = pd.DataFrame(columns=["Actual Returns"])

# Apply linear regression to data with rolling windows
for i in range(0, timeframe):
    
    # Beginning of training window
    start_of_training_period = weeks[i].start_time.strftime(format="%Y-%m-%d")
    
    # End of training window
    end_of_training_period = weeks[training_window+i].end_time.strftime(format="%Y-%m-%d")

    # Window of test-window data
    test_week = weeks[training_window + i + 1]
    
    # String of testing window
    start_of_test_week  = test_week.start_time.strftime(format="%Y-%m-%d")
    end_of_test_week = test_week.end_time.strftime(format="%Y-%m-%d")
    
    train = df.loc[start_of_training_period:end_of_training_period]
    test = df.loc[start_of_test_week:end_of_test_week]
    
    # Create new dataframes
    X_train = train["Lagged_Return"].to_frame()
    y_train = train["Return"]
    X_test = test["Lagged_Return"].to_frame()
    y_test = test["Return"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Create a temporary dataframe
    predictions = pd.DataFrame(predictions, index=X_test.index, columns=["Out-of-Sample Predictions"])
    
    # Create a temporary DataFrame 
    actuals = pd.DataFrame(y_test, index=y_test.index)
    actuals.columns = ["Actual Returns"]  
    
    # Append to dataframes 
    all_predictions = all_predictions.append(predictions)
    all_actuals = all_actuals.append(actuals)   

# Print results
Results = pd.concat([all_actuals, all_predictions], axis=1)
```

<br>


## <u>***Classification***</u>
* Derives categorical conclusions (discrete outcome) based off of classified/modeled data
* The act of discovering whether or not a particular feature or element belongs to a given feature class/group
* Uses a binary (true-positive/true-negative) approach to predict categorical membership (i.e., will the outcome be of type A or type B)


* Examples: `logistic regression`,  `support vector machines`, `neural network`

<br>

#### Model performance

* `Accuracy`, `Precision`, and `Recall` are especially important for classification models


* `Accuracy`: (TP + TN) / (TP + TN + FP + FN)

* `Precision`: TP / (TP + FP)
    - The ratio of correctly predicted positive outcomes out of all predicted positive outcomes
    - High precision relates to a low false-positive rate
    - The question that precision answer is: of all passengers that labeled as survived, how many actually survived?

* `Recall`: TP / (TP + FN)
    - The number of correct positive predictions out of all predictions
    - High recall relates to a low false-negative rate
    - The question recall answers is: Of all the passengers that truly survived, how many did we label?


* `F1 Score`: the weighted average of Precision and Recall
    - F1 is more useful than accuracy when you have an uneven class distribution

    - If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall


* `Classification Report`: includes Accuracy, Precision, Recall, F1-score
```
from sklearn.metrics import classification_report
target_names = ["Class 1", "Class 2"]
print(classification_report(y_test, predictions, target_names=target_names))
```

* `Confusion Matrix`: a table used to describe the performance of the model
    - Columns will reflect the sum of predicted categorical outcomes
    - Rows will reflect the actual sum of outcomes
```
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
```

<br>

## <u>***Logistic Regression***</u>
* Predicts binary outcomes from data
* Employs only linear approach when predicting outcomes
* Supervised learning: in order for the algorithm to learn, it must be given data to learn from
* Centering improves the performance of logistic regression models by ensuring that all data points share the same starting mean value
* Data points with the same starting mean value are clustered together


### 1) Preprocess

* Create random data and visualize it

```
# `make_blobs` creates new random data set
# `centers` helps define the number of classes to create
# `random_state` preserves the state of output
# `random_state` helps ensure the same data set is used to train the model

from sklearn.datasets import make_blobs
X, y = make_blobs(centers=2, random_state=42)

# Visualizing both classes

plt.scatter(X[:, 0], X[:, 1], c=y)
```

* Split into traning and testing data
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
```


### 2) Train

```
# Create a classifier object
# The solver helps optimize learning and computation

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)

# Train the data

classifier.fit(X_train, y_train)

```


### 3) Validate

```
# Score the model

print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
```


### 4) Predict
```
# Predict outcomes for test data set

predictions = classifier.predict(X_test)
pd.DataFrame({"Prediction": predictions, "Actual": y_test})
```


<br>

<br>

## <u>***Support Vector Machines***</u>
* Predicts binary outcomes from data
* Supervised learning: in order for the algorithm to learn, it must be given data to learn from
 * separates classes of data points into multidimensional space which is segmented by a line or hyperplane (a dimensional vector)
 * The goal with hyperplanes is to get the margin of the hyperplane equidistance to the data points for all classes
 * The margin is considered optimal when the distance from the hyperplane and the support vectors are equidistant and as wide as possible
 * The data closest to the margin of the hyperplane are called support vectors, and they are used to define boundaries of the hyperplane
 * Focuses on dimensionality
 * Each feature is a dimension
 * Employs both a linear and non-linear approach when predicting outcomes

 * may introduce a new z-axis dimension for non-linear hyperplanes to establish 0 tolerance with perfect partition

* Kernel is used to identify the orientation of the hyperplane, as either linear or multi-dimensional

* The kernel argument is used to express the degree of dimensionality needed to separate the data into classes

* The decision_function function is used to calculate the classification score for each data point and the values are used to classify the data points to either class


 ```
 from sklearn.svm import SVC
 classifier = SVC(kernel='linear')
 classifier.fit(X_train, y_train)
 predictions = classifier.predict(X_test)
 ```

<br>

### **Compared to Logistic Regression**

 * SVM is more beneficial than Logistic Regression because the model supports the classification of outliers and overlapping data points

 * SVM provides higher accuracy with less computation power


<br>
<br>


## <u>***Tree-base Model***</u>
* Supervised learning mostly used for classification and regression

1. Preprocessing
    * Convert text or categorical data to numerical because machine learning algorithms only works with numerical data

        * The LabelEncoder function performs integer encoding of labels

    * Normalize all input data to the same scale to prevent any single feature from dominating others
        * The StandardScaler function standardizes numerical features
```
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

* 
    * Integer encoding
    * 


    