import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_auth

USERNAME_PASSWORD_PAIRS = [
    ["username", "password"],
    ["Jamesbond", "007"]
]

app = dash.Dash()
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div(
    [
        dcc.RangeSlider(
            id="range-slider",
            min=-5,
            max=6,
            marks={i: str(i) for i in range(-5, 7)},
            value=[-3, 4]
        ),
        html.H1(id="product")
    ],
    style={"width": "50%"}
)


@app.callback(
    Output("product", "children"),
    [Input("range-slider", "value")]
)
def update_value(value_list):
    return value_list[0] * value_list[1]


if __name__ == "__main__":
    app.run()
