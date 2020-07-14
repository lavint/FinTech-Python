# Visualization

## **Matplotlib**
```
import matplotlib.pyplot as plt
```

pyplot graphs
```
fig = plt.figure()

# data.plot.bar(title="Title")
plt.plot(data)

# plt.title("Title", fontdict={"fontsize":20, "fontweight": "bold"})
fig.suptitle("Title", fontsize=20)


plt.xlabel("xlabel", fontsize=18)
plt.ylabel("ylabel", fontsize=16)
plt.ylim([y_min - y_std, y_max + y_std])

fig.savefig('test.jpg')
plt.close(fig)
```

Using axes
```
fig, ax = plt.subplots()   
ax.plot(data)
ax.set_title('Title')
ax.set_xlabel('xlabel', fontsize=18)
ax.set_ylabel('ylabel', fontsize=16)
fig.savefig('test.jpg')
plt.close(fig)    
```

Resources:

[Stack Overflow](https://stackoverflow.com/questions/34162443/why-do-many-examples-use-fig-ax-plt-subplots-in-matplotlib-pyplot-python)

[Matplotlib Documentation](https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.subplots.html)
<br>
<br>

## **HoloViews**
```
import hvplot.pandas
```


HoloViews graphs
```
df.hvplot()

df.hvplot.line()

df.hvplot.scatter(x='A', y='B')

df.hvplot.bar(
    x="ticker", y="daily_return", xlabel="Ticker", ylabel="Daily Return", rot=90
)
```


Plot side by side graphs
```
df1.hvplot.bar() + df2.hvplot.line()
```

Plot in the same graphs
```
df1.hvplot.line() * df2.hvplot.bar()
```

`opts` formatting
```
df.hvplot.bar(x="Date", y="Amt").opts(
    xformatter="%.0f",
    yformatter="%.0f",
    width=1200,
    invert_axes=True,
    bgcolor="lightgray",
    hover_line_color="red",
    line_color='green'
)
```

hvplot `groupby`
* Interactive neighborhood dropdown will be shown automatically 
```
avg_prices_nbh.hvplot.line(
        x = 'year',
        y = 'sale_price_sqr_foot',
        groupby="neighborhood"
    )
```



<br>
<br>

## **Plotly**
```
import plotly.express as px
```



Create scatter plot
```
px.scatter(
    df_housing_sales,
    x="Cost of Living Index",
    y="Average Sale Price",
    size="Number of Housing Units Sold",
    color="County",
)
```



Create area plot
```
px.area(
    df,
    x="date",
    y="price",
    color="state",
    line_group="state",
)
```


Create Parallel Coordinates plot
```
px.parallel_coordinates(df, color='uniqueID')
```



Create Parallel Categories plot
```
px.parallel_categories(
    df,
    dimensions=["type", "region", "prop_size"],
    color="year",
    color_continuous_scale=px.colors.sequential.Inferno,

    # renaming labels
    labels={
        "type": "Type of Dwelling",
        "region": "Region",
        "prop_size": "Property Size",
    },
)

```


Create mapbox plot
```
import os

# Extract token
mapbox_token = os.getenv("MAPBOX_API_KEY")

# Set token using Plotly Express set function
px.set_mapbox_access_token(mapbox_token)


# Plot Data
px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    size="Population",
    color="CityName",
    color_continuous_scale=px.colors.cyclical.IceFire,
    title="City Population",
    zoom=3,
    width=1000
)

```


<br>
<br>

## **Panel**

```
import panel as pn
from panel.interact import interact
from panel import widgets
```

Enable Jupyter Lab Panel plugin
```
pn.extension()                  #hvplot
pn.extension("plotly")          #plotly
```

Drop downs
```
list_of_years = ['2020', '2021', '2022', '2023', '2024']

def choose_year(year):
    return year

interact(choose_year, year=list_of_years)
```


Panel with hvplot
```
# Define function to create plot
def plot_housing(number_of_sales):
    housing_transactions = pd.DataFrame(
        {
            "years": np.random.randint(2020, 2030, number_of_sales),
            "sales": np.random.randint(22, 500, number_of_sales),
            "foreclosures": np.random.randint(10, 176, number_of_sales),
        }
    ).sort_values(["years", "sales"])

    return housing_transactions.hvplot.scatter(
        x="sales",
        y="foreclosures",
        c="years",
        colormap="viridis",
        title="Housing Transactions",
    )


# Render plot with Panel interactive widget ***
interact(plot_housing, number_of_sales=(0, 100))
```



Panel with plotly
```
# Create plot
housing_transactions = pd.DataFrame(
    {
        "years": np.random.randint(2020, 2030, number_of_sales),
        "sales": np.random.randint(22, 500, number_of_sales),
        "foreclosures": np.random.randint(10, 176, number_of_sales),
    }
).sort_values(["years", "sales"])

plot = px.scatter(
    housing_transactions,
    x="sales",
    y="foreclosures",
    color="years",
    title="Housing Transactions",
)


# Wrap Plotly object by explicitly declaring Panel pane ***
pane = pn.pane.Plotly(plot)


# Wrap Plotly object by using panel.panel helper function ***
pn.panel(plot)


# Print the type of object
pane.pprint()

```


Dashboard panel
```
# 2 plots side by side
row = pn.Row(scatter_plot, bar_plot)

# Append to pane
row.append(pie_plot)

# 2 plots up and down
col = pn.Column(scatter_plot, bar_plot)


# Create column using Markdown and row object
column = pn.Column(
    '# Visualizations',
    '## Yeah! Pane is cool',
    row)


# Create tabs
dashboard = pn.Tabs(
    ("Correlations", scatter_plot),
    ("Time Series", bar_plot))
```


<br>

---

**Execute the servable function**
```
# On jupyter lab
dashboard.servable()

# On GitBash
# Navigate to the folder where the ipynb locates
panel serve dashboard_notebook.ipynb
```





