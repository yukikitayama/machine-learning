import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("mpg.csv")

app = dash.Dash()

features = df.columns

app.layout = html.Div(
    [
        html.Div(
            [dcc.Dropdown(
                id="xaxis",
                options=[{"label": col, "value": col} for col in features],
                value="displacement"
            )],
            style={
                "width": "48%",
                "display": "inline-block"
            }
        ),
        html.Div(
            [dcc.Dropdown(
                id="yaxis",
                options=[{"label": col, "value": col} for col in features],
                value="mpg"
            )],
            style={
                "width": "48%",
                "display": "inline-block"
            }
        ),
        dcc.Graph(
            id="feature-graphic"
        )
    ],
    style={
        "padding": 10
    }
)

@app.callback(
    Output(component_id="feature-graphic", component_property="figure"),
    [
        Input(component_id="xaxis", component_property="value"),
        Input(component_id="yaxis", component_property="value")
    ]
)
def update_graph(xaxis_name, yaxis_name):
    data = [
        go.Scatter(
            x=df[xaxis_name],
            y=df[yaxis_name],
            # ?
            text=df["name"],
            mode="markers",
            marker={
                "size": 15,
                "opacity": 0.5,
                "line": {
                    "width": 0.5,
                    "color": "white"
                }
            }
        )
    ]

    layout = go.Layout(
        title="My Dashboard for MPG",
        xaxis={
            "title": xaxis_name,
        },
        yaxis={
            "title": yaxis_name
        },
        hovermode="closest"
    )

    return {
        "data": data,
        "layout": layout
    }


if __name__ == "__main__":
    app.run()
