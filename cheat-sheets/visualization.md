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