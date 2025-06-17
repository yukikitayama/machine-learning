import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import json
import base64


app = dash.Dash()

df = pd.read_csv("wheels.csv")


def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, "rb").read())
    return f"data:image/png;base64,{encoded.decode()}"


app.layout = html.Div(
    [
        html.Div(
            dcc.Graph(
                id="wheels-plot",
                figure={
                    "data": [go.Scatter(x=df["color"], y=df["wheels"], dy=1, mode="markers", marker={"size": 15})],
                    "layout": go.Layout(title="Test", hovermode="closest")
                }
            ),
            style={"width": "30%", "float": "left"}
        ),
        html.Div(
            # html.Pre(
            #     id="hover-data",
            #     style={"paddingTop": 35}
            # ),
            html.Img(
                id="hover-data",
                src="children",
                height=300
            ),
            style={"paddingTop": 35}
        )
    ]
)


@app.callback(
    Output(component_id="hover-data", component_property="src"),
    [Input(component_id="wheels-plot", component_property="hoverData")]
)
def callback_image(hoverData):
    # return json.dumps(hoverData, indent=2)
    wheel = hoverData["points"][0]["y"]
    color = hoverData["points"][0]["x"]
    path = f"images/{df[(df["wheels"] == wheel) & (df["color"] == color)]["image"].values[0]}"
    return encode_image(path)


if __name__ == "__main__":
    app.run()