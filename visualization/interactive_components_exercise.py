import dash
from dash import html, dcc
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div(
    [
        dcc.RangeSlider(
            id="range-slider",
            min=-10,
            max=10,
            marks={i: str(i) for i in range(-10, 11)},
            value=[-1, 1]
        ),
        html.H1(
            id="product"
        )
    ]
)


@app.callback(
    Output(component_id="product", component_property="children"),
    [Input(component_id="range-slider", component_property="value")]
)
def update_value(value_list):
    return value_list[0] * value_list[1]


if __name__ == "__main__":
    app.run()
