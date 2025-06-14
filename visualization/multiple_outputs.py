import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import base64

df = pd.read_csv("wheels.csv")

# print(df.head())

app = dash.Dash()


def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, "rb").read())
    return f"data:image/png;base64,{encoded.decode()}"


app.layout = html.Div(
    [
        dcc.RadioItems(
            id="wheels",
            options=[{"label": i,"value": i} for i in df["wheels"].unique()],
            value=1
        ),
        html.Div(id="wheels-output"),
        html.Hr(),
        dcc.RadioItems(
            id="colors",
            options = [{"label": i, "value": i} for i in df["color"].unique()],
            value = "blue"
        ),
        html.Div(id="colors-output"),
        html.Img(
            id="display-image",
            src="children",
            height=300
        )
    ],
    style={
        "fontFamily": "helvetica",
        "fondSize": 18
    }
)


@app.callback(
    Output(component_id="wheels-output", component_property="children"),
    [Input(component_id="wheels", component_property="value")]
)
def callback_a(wheels_value):
    return f"you chose {wheels_value}"


@app.callback(
    Output(component_id="colors-output", component_property="children"),
    [Input(component_id="colors", component_property="value")]
)
def callback_b(colors_value):
    return f"you chose {colors_value}"


@app.callback(
    Output(component_id="display-image", component_property="src"),
    [
        Input(component_id="wheels", component_property="value"),
        Input(component_id="colors", component_property="value")
    ]
)
def callback_image(wheel, color):
    file_name = df[(df["wheels"] == wheel) & (df["color"] == color)]["image"].values[0]
    path = f"images/{file_name}"
    return encode_image(path)


if __name__ == "__main__":
    app.run()
