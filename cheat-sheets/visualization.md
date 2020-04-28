## **HoloViews ibraries**
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

Opts formatting
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

<br>
<br>

## **Plotly Libraries**
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
)

```


<br>
<br>

## **Panel library**

```
import panel as pn
from panel.interact import interact
from panel import widgets
```

Enable Jupyter Lab Panel plugin
```
pn.extension()
```

Drop downs
```
list_of_years = ['2020', '2021', '2022', '2023', '2024']

def choose_year(year):
    return year

interact(choose_year, year=list_of_years)
```


Panel with plot
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


# Render plot with Panel interactive widget
interact(plot_housing, number_of_sales=100)
```