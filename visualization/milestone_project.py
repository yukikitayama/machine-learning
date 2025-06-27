import os

import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import pandas_datareader.data as web
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

app = dash.Dash()

nsdq = pd.read_csv("NASDAQcompanylist.csv")
nsdq.set_index("Symbol", inplace=True)
options = []
for tic in nsdq.index:
    mydict = {
        "label": nsdq.loc[tic]["Name"] + " " + tic,
        "value": tic
    }
    options.append(mydict)

app.layout = html.Div(
    [
        html.H1("Stock Ticker Dashboard"),
        html.Div(
            [
                html.H3("Enter a stock symbol:", style={"paddingRight": "30px"}),
                dcc.Dropdown(
                    id="my_stock_picker",
                    options=options,
                    value=["TSLA"],
                    multi=True
                )
            ],
            style={"display": "inline-block", "verticalAlign": "top", "width": "30%"}
        ),
        html.Div(
            [
                html.H3("Select a start and end date:"),
                dcc.DatePickerRange(
                    id="my_date_picker",
                    min_date_allowed=datetime(2015, 1, 1),
                    max_date_allowed=datetime.today(),
                    start_date=datetime(2018, 1, 1),
                    end_date=datetime.today()
                ),
            ],
            style={"display": "inline-block"}
        ),
        html.Div(
            [
                html.Button(id="submit-button", n_clicks=0, children="Submit", style={"fontSize": 24, "marginLeft": "30px"})
            ],
            style={"display": "inline-block"}
        ),
        dcc.Graph(
            id="my_graph",
            figure={
                "data": [{"x": [1, 2], "y": [3, 1]}],
                "layout": go.Layout(title="Default title")
            }
        )
    ]
)


@app.callback(
    Output(component_id="my_graph", component_property="figure"),
    [
        Input(component_id="submit-button", component_property="n_clicks")
    ],
    [
        State(component_id="my_stock_picker", component_property="value"),
        State(component_id="my_date_picker", component_property="start_date"),
        State(component_id="my_date_picker", component_property="end_date"),
    ]
)
def update_graph(n_clicks, stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], "%Y-%m-%d")
    end = datetime.strptime(end_date[:10], "%Y-%m-%d")

    traces = []
    for tic in stock_ticker:
        df = web.DataReader(tic, "av-daily", start, end, api_key=os.getenv("ALPHAVANTAGE_API_KET"))
        traces.append({"x": df.index, "y": df["close"], "name": tic})

    fig = {
        "data": traces,
        "layout": go.Layout(title=str(stock_ticker))
    }

    return fig


if __name__ == "__main__":
    app.run()
