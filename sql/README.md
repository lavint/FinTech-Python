# Looking for Suspicious Transactions


Analyze historical credit card transactions and consumption patterns in order to identify possible fraudulent transactions.


1. [Data Modeling](#Data-Modeling):
Define a database model to store the credit card transactions data and create a new PostgreSQL database using your model.

2. [Data Engineering](#Data-Engineering): Create a database schema on PostgreSQL and populate your database from the CSV files provided.

3. [Data Analysis](#Data-Analysis): Analyze the data to identify possible fraudulent transactions.

---

## Files

* [card_holder.csv](Data/card_holder.csv)
* [credit_card.csv](Data/credit_card.csv)
* [merchant_category.csv](Data/merchant_category.csv)
* [merchant.csv](Data/merchant.csv)
* [transaction.csv](Data/transaction.csv)

## Instructions

### **Data Modeling**

Create an entity relationship diagram (ERD) by inspecting the provided CSV files.

**Note:** For the `credit_card` table, the `card` column should be a VARCHAR(20) datatype rather than an INT.

Part of the challenge here is to figure out how many tables to create and the relationships among the tables.

Use `Quick Database Diagrams` [Quick Database Diagrams](https://app.quickdatabasediagrams.com/#/) to createthe ERD model model.

<br>

### **Data Engineering**

Using the database model as a blueprint, create a database schema for each table and their relationships. Specify data types, primary keys, foreign keys, and any other constraints.

After creating the database schema, import the data from the corresponding CSV files.

<br>

### **Data Analysis**

Now that the data is prepared within the database, it's time to identify fraudulent transactions using SQL and Pandas DataFrames. Analyze the data:

* What are the total number of transactions and total spending for each card holder

* What are the 100 highest transactions amount during the time period 7:00 a.m. to 9:00 a.m?

* Some fraudsters hack a credit card by making several small payments (generally less than $2.00), which are typically ignored by cardholders. Count the transactions that are less than $2.00 per cardholder. Is there any evidence to suggest that a credit card has been hacked?

* What are the top five merchants prone to being hacked using small transactions?

* create a view for each of the previous queries.

<br>

Create a report for fraudulent transactions of some top customers of the firm. To achieve this task, perform a visual data analysis of fraudulent transactions using Pandas, Plotly Express, hvPlot, and SQLAlchemy to create the visualizations.

* Verify if there are any fraudulent transactions in the history of two of cardholders' IDs 18 and 2.

  * Using hvPlot, create a line plot representing the time series of transactions over the course of the year for each cardholder. In order to compare the patterns of both cardholders, create a line plot containing both lines.

  * What difference do you observe between the consumption patterns? Does the difference suggest a fraudulent transaction? Explain your rationale.

* The CEO of the firm's biggest customer suspects that someone has used her corporate credit card without authorization in the first quarter of 2018 to pay for several expensive restaurant bills. You are asked to find any anomalous transactions during that period.

  * Using Plotly Express, create a series of six box plots, one for each month, in order to identify how many outliers there are per month for cardholder ID 25.

  * Do you notice any anomalies? Describe your observations and conclusions.

## Challenge

Another approach to identify fraudulent transactions is to look for outliers in the data. Standard deviation or quartiles are often used to detect outliers.

Read the following articles on outliers detection, and then code a function using Python to identify anomalies for any cardholder.

* [How to Calculate Outliers](https://www.wikihow.com/Calculate-Outliers)

* [Removing Outliers Using Standard Deviation in Python](https://www.kdnuggets.com/2017/02/removing-outliers-standard-deviation-python.html)

* [How to Use Statistics to Identify Outliers in Data](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)



## Note

For comparing time and dates, take a look at the [date/time functions and operators](https://www.postgresql.org/docs/8.0/functions-datetime.html) in the PostgreSQL documentation.


