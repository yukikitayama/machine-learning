import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import requests

app = dash.Dash()

app.layout = html.Div(
    [
        html.Div(
            [
                html.Iframe(src="https://www.flightrader24.com", height=500, width=1200)
            ]
        ),
        html.Div(
            [
                html.Pre(id="counter_text", children="Active Flights Worldwide"),
                dcc.Interval(id="interval-component", interval=6000, n_intervals=0)
            ]
        )
    ]
)


@app.callback(
    Output(component_id="counter_text", component_property="children"),
    [Input("interval-component", "n_intervals")]
)
def update_layout(n):
    url = "https://data-live.flightradar24.com/zones/fcgi/feed.js?faa=1\
           &mlat=1&flarm=1&adsb=1&gnd=1&air=1&vehicles=1&estimated=1&stats=1"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    data = res.json()
    counter = 0
    for element in data["stats"]["total"]:
        counter += data["stats"]["total"][element]
    return f"Active flights worldwide: {counter}"


if __name__ == "__main__":
    app.run()
