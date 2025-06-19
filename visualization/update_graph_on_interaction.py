import dash
from dash import html, dcc
import plotly.graph_objs as go
import pandas as pd
import numpy as np

app = dash.Dash()

df = pd.read_csv("mpg.csv")
df["year"] = np.random.randint(-4, 5, len(df)) * 0.1 + df["model_year"]

app.layout = html.Div(
    [
        dcc.Graph(
            id="mpg-scatter",
            figure={
                "data": [
                    go.Scatter(
                        x=df["year"] + 1900,
                        y=df["mpg"],
                        text=df["name"],
                        hoverinfo="text+y+x",
                        mode="markers"
                    )
                ],
                "layout": go.Layout(
                    title="MPG Data",
                    xaxis={"title": "Model Year"},
                    yaxis={"title": "MPG"},
                    hovermode="closest"
                )
            }
        )
    ]
)


if __name__ == "__main__":
    app.run()
