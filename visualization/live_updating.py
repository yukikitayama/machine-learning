import dash
from dash import html, dcc
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div(
    [
        html.H1(id="live-update-text"),
        dcc.Interval(id="interval-component", interval=2000, n_intervals=0)
    ]
)


@app.callback(
    Output(component_id="live-update-text", component_property="children"),
    [Input(component_id="interval-component", component_property="n_intervals")]
)
def update_layout(n):
    return f"Crash free for {n} refreshes"

if __name__ == "__main__":
    app.run()
