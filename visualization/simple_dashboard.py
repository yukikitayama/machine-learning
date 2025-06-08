import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("OldFaithful.csv")

app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Graph(
            id="old_faithful",
            figure={
                "data": [
                    go.Scatter(
                        x=df["X"],
                        y=df["Y"],
                        mode="markers"
                    )
                ],
                "layout": go.Layout(
                    title="Old Faithful Eruptions",
                    xaxis={
                        "title": "Duration"
                    },
                    yaxis={
                        "title": "Interval"
                    }
                )
            }
        )
    ]
)

if __name__ == "__main__":
    app.run()
