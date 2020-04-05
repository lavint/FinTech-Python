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
df.describe()
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



Check number of rows of data for each column
```
df.count()  
```



Check frequency of a value in a column
```
df['col1'].value_counts()
```



Working with nulls
* `Inplace = True` transforms the data in place
* `Inplace = False` performs the operation and returns a copy of the object         
```
df.isnull()                     # Check for nulls for each column

df.isnull().mean() * 100        # Calculate the percentage of nulls per column

df.isnull().sum()               # Calculate the sum of nulls per column

df['col1'].fillna('N/A', inplace = True)    # Fill nulls with a value in a column
df['col1'] = df['col1'].fillna('N/A')               # Same as above

df.dropna(inplace = True)       # Drop rows that contain at least 1 null
```



Working with duplicates
```
df.duplicated()         # Check duplicates on Index, return True or False

df['col1'].duplicated() # Check duplicates on a column, return True or False

df.drop_duplicates()    # Keep first occurrence, drop all duplicates (match entire row)

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



Output DataFrame to CSV file
```
df.to_csv('output.csv')
```





