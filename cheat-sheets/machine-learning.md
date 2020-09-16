# Machine Learning

Instead of having to configure inputs and manually make changes to an algorithm, machine learning programs automatically adapt to improve outcomes and predictions, as well as accuracy and precision.

The statistical or algorithmic model of the data can be used to make predictions or decisions about new data automatically.

For example, instead of creating some if-else decision structure in order to identify if a transaction is fraudulent, a machine learning algorithm can review all transactions ever made by an account owner, classify and cluster transactions,and then predict whether or not a transaction is fraudulent.

## <u>***Pipeline***</u>

<br>

* Data selection: What data is available, what data is missing, and what data can be removed
* Data preprocessing: Organize the selected data by formatting, cleaning, and sampling it
* Data transformation: Transform the data to a format that eases its treatment and storage for future use (e.g., CSV file, spreadsheet, database)

<br>

1) Preprocess/Clean
    
    z) Remove nulls ot duplicates 

    a) Split data into X, y
    
    b) Encode categorical features X
    
    c) Split data into X_train, X_test, y_train, y_test
    
    d) Scale X_train, X_test by using the scaler that's trained with X_train (CRITICAL to have features on the same scale)
    
    e1) Take care of imbalanced data (oversampling or undersampling) using X_train_scaled, y_train
    
    e2) Skip this step and use classifiers in `imblearn.ensemble` which automatically take care of imbalance issue 

<br>

2) Train/Fit
    
    a) Pick the model and create a model instance
    
    b) Train the model with X_resampled, y_resampled

<br>

3) Validate (if you've split the data into 3 sets)

<br>

4) Predict

    a) predict y_pred using X_test_scaled

<br>

5) Check accuracy

    a) compare y_pred and y_test using different matrix
    


<br>
<br>

## <u>***Intelligent algorithms***</u>
* Use pre-existing data to learn and make decisions on how to configure and adapt its behavior for the most accurate and precise prediction
* Are used to fuel machine learning, predictive analytics, and artificial intelligence


### *Supervised vs Unsupervised*

<br>


| Supervised Learning                         |	Unsupervised Learning                   |
| ---------                                   | -----------                             |
| Input data is labeled                       | Input data is unlabeled                 |
| Potential outcomes need to be known upfront | Learns on the fly without having seen any type of data before |
| Uses training datasets                      |	Uses just input datasets                |
| Predict a class or value based on labeled historical data | Determine patterns or group data by clustering data|
| classification, regression                  | dimensionality reduction, clustering |


<br>
<br>

## <u>***Predictive Analytics***</u>
* Machine learning is a component of predictive analytics

<br>

## <u>***Artificial Intelligence***</u>
* Machine learning is an application of AI: machines can execute tasks and learn while doing so in order to perform more intelligently
* Instead of programming machines to perform specific tasks, program machines to learn what tasks to complete and how to complete them
* (Deep Learning is also an application of AI)

<br>
<br>

# Time Series

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
<br>

# Supervised Learning

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


* When the classes are imbalanced, the precision and recall for one class is greater than the other class even though the accuracy score is high

* When false positives and false negatives are weighted fairly evenly, accuracy or F1 score is effective to compare models

* `Accuracy`: (TP + TN) / (TP + TN + FP + FN)
```
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, predictions)
```


* `Precision`: TP / (TP + FP)
    - The ratio of correctly predicted positive outcomes out of all predicted positive outcomes
    - High precision relates to a low false-positive rate
    - The question that precision answer is: of all passengers that labeled as survived, how many actually survived?
    - When false positives are more costly than false negatives, precision is effective to compare models

* `Recall`: TP / (TP + FN)
    - The number of correct positive predictions out of all predictions
    - High recall relates to a low false-negative rate
    - The question recall answers is: Of all the passengers that truly survived, how many did we label?
    - When false negatives are more costly than false positives, recall is effective to compare models


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

cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)
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


## <u>***Categorical Data Preprocessing***</u>

***Convert text or categorical data to numerical***

* Because machine learning algorithms only works with numerical data

* The LabelEncoder function performs integer encoding of labels


```
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

I) Encode using LabelEncoder()
```
# Create encoding instance
label_encoder = LabelEncoder()

# train encoder to learn how many classes to use for encoding
label_encoder.fit(df["month"])

# Check classes identified by encoder
list(label_encoder.classes_)

# Transform month from text to integer
df["month_int"] = label_encoder.transform(df["month"])
```

* Certain machine learning models may place numerical significance on integer encodings (higher number, more significant); Thus, the above 2 methods are not ideal in that case.


II) Encode manually
```
months_num = {
"January": 1,
"February": 2,
"March": 3,
"April": 4,
"May": 5,
"June": 6,
"July": 7,
"August": 8,
"September": 9,
"October": 10,
"November": 11,
"December": 12,
}

# Same as above and reduce manual work
months_num = {name: num for num, name in enumerate(calendar.month_name) if num}

# Map the dictionary values to each value in the series using dictionary keys
df["month_num"] = df["month"].apply(lambda x: months_num[x])
``` 

III) Encode using pd.get_dummies() - binary result
```
binary_encoded = pd.get_dummies(df, columns=["gender"])
```
<br>


***Normalize all input data to the same scale***
*  To prevent any single feature from dominating others
* It is always a good idea to have features all on the same scale, so they have equal importance to the model
* The StandardScaler function standardizes numerical features


I) `MinMaxScaler` scales data between 0 and 1

II) `StandardScaler` standardizes features by removing the mean and scaling to unit variance


```
# Creating the scaler instance
data_scaler = StandardScaler()

# Fitting the scaler
data_scaler.fit(loans_binary_encoded)

# Transforming the data
data_scaled = data_scaler.transform(binary_encoded)
```


<br>


## <u>***Tree Base Model***</u>
* Supervised learning mostly used for classification and regression
* Examples: decision trees, random forests, gradient boosting trees


## **Decision Trees**
* Encode a series of True/False questions which can be represented by if/else statements
* Deep and complex trees tend to overfit data and do not generalize well

```
from sklearn import tree
import pydotplus
from IPython.display import Image

# Creating the decision tree classifier instance
model = tree.DecisionTreeClassifier()

# Fitting the model
model = model.fit(X_train_scaled, y_train)

# Making predictions using the testing data
predictions = model.predict(X_test_scaled)
```

### **Visualize trees**

```
# Create DOT data
X = df.copy()
dot_data = tree.export_graphviz(
    model, out_file=None, feature_names=X.columns, class_names=["0", "1"], filled=True
)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())
```

<br>

## <u>***Ensemble Learning***</u>
*  Combines many weak learners (result of too few features or data points) to create a more accurate and roboust prediction engine

* Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees

* Supervised learning

* Examples: Random Forests, GradientBoostedTree, XGBoost, Boosting, Bagging


<br>

## **Random Forests**
* Each tree is much simpler because it is built from a subset of the data by randomly sampling

* Is robust against overfitting because all of those weak classifiers are trained on different pieces of the data

* Is also robust to outliers and non-linear data by binning them

* Runs efficiently on large databases

* Handles thousands of input variables without variable deletion

* Is used to rank the importance of input variables in a natural way

```
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
# Range between 64 and 128 trees is recommended for initial modeling

rf_model = RandomForestClassifier(n_estimators=500, random_state=78)


# Fitting the model
rf_model = rf_model.fit(X_train_scaled, y_train)

# Making predictions using the testing data
predictions = rf_model.predict(X_test_scaled)


# Random Forests in sklearn will automatically calculate feature importance
importances = rf_model.feature_importances_


# Sort the features by their importance
X = df.copy()
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
```

 * To improve a random forests model, we can:
    * Reduce the number of features using PCA.
    * Create new features based on new data from the problem domain.
    * Increase the number of estimators.


<br>

## **Boosting**
* Takes multiple algorithms and coordinates them as an ensemble and runs the algorithms iteratively to identify the best prediction

* Takes the predictions of each weak learner and aggregate them to produce a more accurate and precise prediction

* Both a process and a set of meta-algorithms hat are used to improve the performance of weak learners

* Works with and affects other algorithms, not the data

* Uses weighted averages (the higher the average, the more inaccurate the prediction) to determine what values are misclassified

* Weighs predictions based on accuracy - as long as data points are weighted as inaccurate, boosting algorithms will continue to resample with greater frequency those samples that previously had the highest error

<br>

## **Bagging**
* Focuses on re-sampling data and running with different models on the fly in order to formulate the most accurate and precise prediction

* Improves the accuracy and robustness of a model

* Each classifier runs independently of the others

* Once all classifiers are finished predicting, the bagging algorithm will aggregate results via voting process

* Each classifier will vote for a label, and then the bagging algorithm will aggregate votes and classify the label with the most votes as the prediction

* Instead of weighing predictions, bagging algorithms resample and replace data, and combine prediction from multiple models


<br>

## **Gradient Boosted Tree**

* Combine weak learners together and executes them in parallel in order to refit the model as needed

* All the weak learners in a gradient boosting machine are decision trees

```
from sklearn.ensemble import GradientBoostingClassifier
```

* `n_estimators` determines the number of weak learners to use

* The higher the value of `n_estimators`, the more trees that will be created to train the algorithm. The more trees, the better the performance but the slower the model runs

* `max_depth` identifies the size/depth of each decision tree

* `Learning rate` identifies how aggressive the algorithm will learn and controls overfitting. Smaller values should be used when setting learning_rate. Learning_rate will work with n_estimators to identify the number of weak learners to train


* To determine the optimal `learning rate`, pick the one with the highest proportion of training and testing accuracy
```
# Create a classifier object
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    classifier = GradientBoostingClassifier(
    n_estimators=20,
    learning_rate=learning_rate,
    max_features=5,
    max_depth=3,
    random_state=0
    )

    # Fit the model
    classifier.fit(X_train_scaled, y_train.ravel())
    print("Learning rate: ", learning_rate)

    # Score the model
    print("Accuracy score (training): {0:.3f}".format(classifier.score(X_train_scaled, y_train.ravel())))
    print("Accuracy score (validation): {0:.3f}".format(classifier.score(X_test_scaled, y_test.ravel())))
    print()
```

* Determine the learning rate the model uses
```
# Choose a learning rate and create classifier
classifier = GradientBoostingClassifier(
    n_estimators=20,
    learning_rate=0.75,
    max_features=5,
    max_depth=3,
    random_state=0
)

# Fit the model
classifier.fit(X_train_scaled, y_train.ravel())

# Make Prediction
predictions = classifier.predict(X_test_scaled)
pd.DataFrame({"Prediction": predictions, "Actual": y_test.ravel()}).head(20)
```

<br>

## <u>***Dealing With Imbalanced Classes***</u>
* Imbalanced Classes: Model will be better at predicting the majority class because model fitting algorithms are designed to minimize the number of total incorrect classifications

### **Solutions**

* Prior to the oversampling/undersampling, we will split up the data into training and test sets as we normally do. This is because even though we want the training set to be oversampled to account for imbalance, we should always make sure that the test set to be "real"

    ```
    # Train test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    ```



* **Oversampling**: Pick more data points from minority class for training

    
    1. Random

        * Replicates the existing training set, randomly choosing additional instances of the minority class with replacement until the minority class is equal to the majority class in size

        * Is more likely to create overfitting problems  due to the lack of variation in the repeated instances


        ````
        # Oversample the training set with RANDOM
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter

        ros = RandomOverSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        Counter(y_resampled)


    2. SMOTE (Synthetic Minority Oversampling Technique)

        * Generates synthetic data by first identifying cluster centers in the minority data and then randomly introducing variations to those centers to create new instances

        * Can create noisy dataset when it creates new points that are heavily influenced by outliers

        * Samples are mostly artificial can actually decrease model performance if the generated data does not have the same structure as the observed data does


        ```
        # Oversampling with SMOTE
        from imblearn.over_sampling import SMOTE

        X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
            X_train, y_train
        )
        ```


<br>

* **Undersampling**: Reduces data points from majority class to match the number of minority class

    * Is practical only when there is enough data in the training set

        1. Random

        ```
        from imblearn.under_sampling import RandomUnderSampler

        ros = RandomUnderSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        ```


        2. Cluster Centroid

        * The algorithm first creates n clusters in the majority class training data using the K-means clustering strategy, where n is equal to the number of minority class training instances, and then takes the centroids of those clusters to be the majority class training set.


        ```
        from imblearn.under_sampling import ClusterCentroids

        cc = ClusterCentroids(random_state=1)
        X_resampled, y_resampled = cc.fit_resample(X_train, y_train)

        from collections import Counter

        Counter(y_resampled)

        ```


<br>

* **Combination Sampling**: Oversampling + Undersampling

    1. SMOTEENN (edited-nearest-neighbor)

        * Looks at the labels for the sampled data and removes instances that are surrounded by data points of the other class

        * Prunes data points that are noisy

        ```        
        #SMOTEENN combination sampling
        from imblearn.combine import SMOTEENN

        sm_enn = SMOTEENN(random_state=1)
        X_resampled, y_resampled = sm_enn.fit_resample(X_train, y_train)
        
        
        # Plot data
        plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
        ```


<br>

* **After preprocessing the training sample in imbalanced dataset, you can use ML models on the new X_train, y_train**


<br>

* **Check accuracy for imbalanced data set**

    ```
    # Check accuracy
    from sklearn.metrics import balanced_accuracy_score
    from imblearn.metrics import classification_report_imbalanced
    from sklearn.metrics import confusion_matrix

    print(balanced_accuracy_score(y_test, y_pred))
    print(classification_report_imbalanced(y_test, y_pred))

    y_pred = model.predict(X_test)
    confusion_matrix(y_test, y_pred)
    ```    



<br>

## <u>***Pecision-Recall Curve***</u>
* Is used for comparing multiple models

* An increase in precision (the % of predicted positives that are classified correctly) leads to a fall in recall (the % of actually true positives that are classified correctly)

* A greater area under the PR curveis the superior model

```
from sklearn.metrics import precision_recall_curve

probs_lr = model.predict_proba(X_test)[:, 1]
probs_rf = brf.predict_proba(X_test)[:, 1]
precision_lr, recall_lr, _ = precision_recall_curve(y_test, probs_lr, pos_label=1)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf, pos_label=1)

plt.plot(recall_lr, precision_lr, marker='.')
plt.plot(recall_rf, precision_rf, marker='x')
```



<br>
<br>

# Unsupervised Learning

* No target variable `y`

<br>
Two main applications:

1) Clustering
    * Allows us to split the dataset into groups according to similarity automatically
    * For example, customer segmentation based on buying habits, needs, etc


2) Anomaly detection
    * Automatically discovers unusual data points in a dataset
    * For example, fraudulent transactions identification



<br>

## <u>***K-Means Clustering***</u>

* Groups data into `k` clusters, where each piece of data is assigned to a cluster based on some similarity or distance measure to a `centroid`
   
* A `centroid` represents a data point that is the arithmetic mean position of all the points on a cluster

* How it works:

    1. Randomly initialize the k starting centroids
    2. Each data point is assigned to its nearest centroid
    3. The centroids are recomputed as the mean of the data points assigned to the respective cluster
    4. Repeat steps 1 through 3 until the stopping criteria is triggered


<br>


* Find the best number for k

    * Use `inertia`: The k value where adding more clusters only marginally decreases the inertia (the sum of squared distances of samples to their closest cluster center)
    * Use `elbow curve` (visual of inertia): The k value where the curve turns like an elbow

    ```
    inertia = []
    k = list(range(1, 11))

    # Calculate the inertia for the range of k values
    # Usually 10 is a good number to start

    for i in k:
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(df_shopping)
        inertia.append(km.inertia_)

    # Create the Elbow Curve using hvPlot

    elbow_data = {"k": k, "inertia": inertia}
    df_elbow = pd.DataFrame(elbow_data)
    df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")
    ```

<br>

* Create a function to find the k clusters using K-Means on data and return a dataframe with features and the prediction

    ```
    def get_clusters(k, data):

        # Initialize the K-Means model
        model = KMeans(n_clusters=k, random_state=0)

        # Fit the model
        model.fit(data)

        # Predict clusters
        predictions = model.predict(data)

        # Create return DataFrame with predicted clusters
        data["class"] = model.labels_

        return data

    ```


<br>

* Analyze result visually
    
    ```
    best_clusters = get_clusters(5, df)
    ```

    2D:
    ```
    best_clusters.hvplot.scatter(x="feature A", y="feature B", by="class")

    ```


    3D:
    ```
    fig = px.scatter_3d(
        best_clusters,
        x="feature A",
        y="feature B",
        z="feature C",
        color="class",
        symbol="class",
        width=800,
        )
    fig.update_layout(legend=dict(x=0, y=1))
    fig.show()
    ```



<br>

* Naming classes

    * K-means algorithm is only able to identify how many clusters are in the data and label them with numbers
    * Need subject matter experts to identify the number representation