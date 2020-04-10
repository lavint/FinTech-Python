# Pandas Cheat Sheet

<img src="Images\HiPanda.jpg" alt="drawing" width="600"/>

-----

Import Pandas library
```
import pandas as pd
```



Read CSV into DataFrame
```
df = pd.read_csv(csvpath)                   # Read the first row as the header by default

df = pd.read_csv(csvpath, header=None)      # Read the first row as data

df = pd.read_csv(csvpath, index_col='col2') # Read col2 as the index and not a column

df = pd.read_csv(csv_path, index_col='Date', parse_dates=True, infer_datetime_format=True)

```



Get subsets from the DataFrame
```
df.head()   # first 5 rows by default

df.head(23) # first 23 rows

df.tail()   # bottom 5 rows by default

df.sample(5)    # Random 5 rows

```



Output the shape of DataFrame
```
df.shape
```



Check and change types in DateFrame
```
df.dtypes

df['col1'].dtype

df['col1'] = df['col1'].astype('float')     # change type to float, no inplace option
```



Generate summary statistics:
```
df.describe()                   # Show only numerical columns by default
df.describe(include='all')      # Include all columns
```


Copy DataFrame
* Pandas .copy() method is used to create a copy of a Pandas object. Variables are also used to generate copy of an object but variables are just pointer to an object and any change in new data will also change the previous data.
```
new_df = df.copy()
```



Set index
```
df.set_index(df['col1'], inplace=True)
```

Set date as index
```
df.set_index(pd.to_datetime(df['Date'], infer_datetime_format=True), inplace=True)
```



Sort 
```
df.sort_index(inplace=True)     # Sort index ascending

df.sort_values('col1', inplace=True)          # Sort values ascending
df.sort_values(['col1', 'col2'], ascending = [True, False], inplace=True)
```


Get top 20 items and keep only the first occurrence when duplicates
```
top20 = df.nlargest(20, 'col1', keep='first')
```



Index using `iloc` which gets rows (or columns) at particular positions in the index
```
df.iloc[0]              # Select first row
df.iloc[0:2]            # Select first 2 rows

df.iloc[:,0].head()     # Select all rows and first column, return top 5 results
df.iloc[:,0:2].head()   # Select all rows and first 2 columns

df.iloc[[1,3],[1,6]]    # Select 2nd & 4th rows, 2nd and 7th columns


df.columns.get_loc('col1')                              # Search for column index of 'col1'
df.iloc[0, df.columns.get_loc('col1')] = 'new_value'    # Modify 'col1' value of the first row


df.iloc[(df['col1'] >= 50).values, [1, 3]]      # Filter with iloc using numpy array
                   
```



Index using `loc` which gets rows (or columns) with particular labels from the index
```
df.loc['Chris']                 # Select the row with the index 'Chris'
df.loc['Chris':'Lav']           # Select a range of outputs based on index


df.loc['Lav', 'col1'] = 'new_value'         # Modify 'col1' value of row with index 'Lav'


pd.options.mode.chained_assignment = None  # default='warn'
OR 
make a copy of the original df and then slice the new df


df.loc[df['col1'] == 'old', 'col2'] = new_value     # Change row values in 'col2' to 'new_value' 
                                                    # whenever column 'col1' has value 'old'

df.loc[df['col1'] == 'old', 'new_col'] = new_value  # Create a 'new_col' with 'new_value' 
                                                    # whenever column 'col1' has value 'old'


df.loc[df['col2'] >= 50]                                    # Filter with loc using boolean series
df.loc[(df['col2'] >= 50) & (df['col3'] == 100), 'col1']    # And operation, return 'col1' column only
df.loc[(df['col2'] >= 50) | (df['col3'] == 100), 'col1']    # Or operation, return 'col1' column only
```



Drop column
```
df.drop('col1', axis=1, inplace=True)
df.drop(columns=['col1', 'col2'], inplace=True)
```



Rename all columns (The length of new_col_names must match the number of columns in df)
```
new_col_names = [new_name1, new_name2, new_name3]
df.columns = new_col_names
```



Rename certain columns
```
df = df.rename(columns={
    "col1": "new_name1",
    "col5": "new_name5"
})
```



Manipulate the DataFrame (Assuming 5 columns in df to begin with)
```
df = df[['col5', 'col4', 'col1', 'col2', 'col3']]   # Reorder columns

df = df[['col1', 'col3', 'col4']]                   # Redeclare df with only col1,col3,col4 

df['col6'] = df['col3'] + df['col4']                # Create columns

df = df.drop(columns=['col6'])                      # Delete columns
```



Count number of rows of data for each column
```
df.count()  
```



Calculate unique values and counts for a column
```
df['col1'].value_counts()
```



Get a list of unique values in a column
```
df['col1'].unique().tolist()
```



Count unique values in a column
```
len(df['col1'].unique().tolist())
df['col1'].nunique()
```



Working with nulls
* `Inplace = True` transforms the data in place
* `Inplace = False` performs the operation and returns a copy of the object         
```
df.isnull()                     # Check for nulls for each column

df.isnull().mean() * 100        # Calculate the percentage of nulls per column

df.isnull().sum()               # Calculate the sum of nulls per column

df['col1'].fillna('N/A', inplace=True)    # Fill nulls with a value in a column
df['col1'] = df['col1'].fillna('N/A')     # Same as above

df.dropna(inplace=True)       # Drop rows that contain at least 1 null
```



Working with duplicates
```
df.duplicated()         # Check duplicates on Index, return True or False

df['col1'].duplicated() # Check duplicates on a column, return True or False

df.drop_duplicates()    # Keep first occurrence, drop all duplicates (when match entire row)

df.drop_duplicates(subset=['col2'])     # base only on col2 ******************

df[df['col1'].duplicated() == True]     # Return the row that has duplicate value in col1



```



Split string columns
* Use `expand=True` to split strings into separate columns
* If `True`, returns a DataFrame/MultiIndex. If `False`, returns a Series/Index containing lists of strings
```
split_name_df = people_df['full_name'].str.split(' ', expand=True)      # Split by space

people_df['first_name'] = split_name_df[0]      # Create new column 'first_name' in people_df
people_df['last_name'] = split_name_df[1]       # Create new column 'last_name' in people_df
```



Replace strings
```
df['price'] = df['price'].str.replace('$', '')      # No inplace option
df['price'].str.startswith('$').sum()               # Count strings start with $
```



Convert string Date time into Python Date time object
```
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
```



Output DataFrame to CSV file
```
df.to_csv('output.csv')
```



Visualization
```
%matplotlib inline      # Show plots and store them within the notebook

df.plot()               # Plot a line chart
df.plot(legend=True)
df.plot(kind='bar', figsize=(20,10))

df['col1'].value_counts().plot(kind='pie')                              # Same
df['col1'].value_counts().plot(kind='pie', labels=df['col1'].values)    # Same


df.plot(kind='scatter', x = 'col1', y ='col2')

df.plot.hist(stacked=True, bins=100)    # Plot stacked histogram
df.plot.box()                           # Plot box plot

```







Grouping with aggregate function
```
df.groupby('col1').count()      # Calculate count

df.groupby('col1').mean()       # Calculate average

df.groupby('col1')['col2'].plot()

df.groupby(['col1', 'col2'])['col2'].count()

rounded_df = df.round({'col1':2})

```


Multi-Indexing groupby
* The `first()` / `last()` function is used to subset initial / final periods of time series data based on a date offset
```
df = pd.read_csv(csv_path, parse_dates=True, index_col='Date', infer_datetime_format=True)
df = df.groupby([df.index.year, df.index.month, df.index.day]).first()

df = df.loc[2020,4,6]   # Slice data with date index
```


Concat data by rows
```
joined_df = pd.concat([df1, df2, df3], axis='rows', join='inner')

joined_df = pd.concat([df1, df2, df3], axis=0, join='inner')

df1.merge(df2, how='inner', left_index=True, right_index=True, suffixes = ('_a', '_b'))
# Whenever df1 and df2 have the same column name, insert suffixes for each column
```
<br>
<br>
Inplace method overwrites the original df 

df = filtered_df method returns a copy so you need to store it to a variable

The inplace parameter is commonly used with the following methods:
* `dropna()`
* `drop_duplicates()`
* `fillna()`
* `query()`
* `rename()`
* `reset_index()`
* `sort_index()`
* `sort_values()`

<br>

# Quantitative Analysis + Coding

Calculate returns
```
daily_returns = (price_df - price_df.shift(1)) / price_df.shift(1)
daily_returns = price_df.pct_change()


weekly_returns = (price_df - price_df.shift(7)) / price_df.shift(7)
weekly_returns = price_df.pct_change(7)

cumulative_returns = (1 + daily_returns).cumprod()

```

Std
```
daily_std = daily_returns.std()
annualized_std = daily_std * np.sqrt(252)       # 252 trading days
```



volatility
```
volatility = all_returns.std() * np.sqrt(252)
```



Sharpe ratio
```
# Calculate daily returns
portfolio_a_returns = portfolio_a.pct_change().dropna()
portfolio_b_returns = portfolio_b.pct_change().dropna()

# Concat returns into one DataFrame
all_pfl_returns = pd.concat([portfolio_a_returns, portfolio_b_returns], axis='columns', join='inner')

# Calculate Sharpe Ratio
sharpe_ratios = (all_pfl_returns.mean() * 252) / (all_pfl_returns.std() * np.sqrt(252))


# Average Sharpe Ratio
sharpe_ratios.mean()

```


Correlation
* The Pearson Correlation (which is R, not R-squared) is an indication of the extent of the linear relationship between 2 targets
```
correlation = combined_df.corr()

import seaborn as sns
sns.heatmap(correlation, vmin=-1, vmax=1)

# vmin = min limit; vmax = max limit

daily_returns.corr(method="pearson")
```


Covariance
```
daily_returns = combined_df.pct_change()
covariance = daily_returns['AMZN'].cov(daily_returns['S&P 500'])
```



Variance
```
daily_returns = combined_df.pct_change()
variance = daily_returns['S&P 500'].var()
```



Beta -  a measure of volatility relative to the market
* a Beta of 1.3 is approximately 30% more volatile than the market
```
amzn_beta = covariance / variance
```



Rolling Stats
```
df.rolling(window=7).mean().plot()
df.rolling(window=30).std().plot()

# Set figure of the daily prices of df
ax = df.plot()

# Plot 180-Day Rolling Mean on the same figure
df.rolling(window=180).mean().plot(ax=ax)

# Set the legend of the figure
ax.legend(["df", "df 180 Day Mean"]);


rolling_covariance = daily_returns['AMZN'].rolling(window=30).cov(daily_returns['S&P 500'])

rolling_variance = daily_returns['S&P 500'].rolling(window=30).var()

rolling_beta = rolling_covariance / rolling_variance

rolling_beta.plot(figsize=(20, 10), title='Rolling 30-Day Beta of AMZN')

import seaborn as sns
sns.lmplot(x='S&P 500', y='AMZN', data=daily_returns, aspect=1.5, fit_reg=True)
```



Portfolio Returns
```
# Calculate Portfolio Returns with an equal amount of each stock

initial_investment = 10000

weights = [0.5, 0.5]

portfolio_returns = all_returns.dot(weights)

cumulative_returns = (1 + portfolio_returns).cumprod()

(initial_investment * cumulative_returns).plot()
```


More on pandas
```
joined_df = joined_df.reset_index()

joined_df = joined_df.pivot_table(values="Price", index="Date", columns="Symbol")
```