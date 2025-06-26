import os

import dash
from dash import html, dcc
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import pandas_datareader.data as web
from datetime import datetime

os.environ["ALPHAVANTAGE_API_KEY"] = "2RT66I5C9HNGLKRS"

app = dash.Dash()

app.layout = html.Div(
    [
        html.H1("Stock Ticker Dashboard"),
        html.Div(
            [
                html.H3("Enter a stock symbol:", style={"paddingRight": "30px"}),
                dcc.Input(
                    id="my_stock_picker",
                    value="TSLA",
                    style={"fondSize": 24, "width": 75}
                )
            ],
            style={"display": "inline-block", "verticalAlign": "top"}
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
        Input(component_id="my_stock_picker", component_property="value"),
        Input(component_id="my_date_picker", component_property="start_date"),
        Input(component_id="my_date_picker", component_property="end_date"),
    ]
)
def update_graph(stock_ticker, start_date, end_date):
    start = datetime.strptime(start_date[:10], "%Y-%m-%d")
    end = datetime.strptime(end_date[:10], "%Y-%m-%d")
    df = web.DataReader(stock_ticker, "av-daily", start, end, api_key=os.getenv("ALPHAVANTAGE_API_KET"))
    fig = {
        "data": [{"x": df.index, "y": df["close"]}],
        "layout": go.Layout(title=stock_ticker)
    }
    return fig


if __name__ == "__main__":
    app.run()
