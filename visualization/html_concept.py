import dash
from dash import html

app = dash.Dash()

app.layout = html.Div(
    children=[
        "This is the outermost div!",
        html.Div(
            ["This is an inner div!"],
            style={
                "color": "red",
                "border": "2px red solid"
            }
        ),
        html.Div(
            ["Another inner div!"],
            style={
                "color": "blue",
                "border": "3px blue solid"
            }
        )
    ],
    style={
        "color": "green",
        "border": "2px green solid"
    }
)

if __name__ == "__main__":
    app.run()