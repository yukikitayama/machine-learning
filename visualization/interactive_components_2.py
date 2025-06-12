import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

df = pd.read_csv("gapminderDataFiveYear.csv")

app = dash.Dash()

year_options = []
for year in df["year"].unique():
    year_options.append({"label": str(year), "value": year})

app.layout = html.Div(
    [
        dcc.Graph(id="graph"),
        dcc.Dropdown(
            id="year-picker",
            options=year_options,
            value=df["year"].min()
        )
    ]
)


@app.callback(
    Output(component_id="graph", component_property="figure"),
    [Input(component_id="year-picker", component_property="value")]
)
def update_figure(selected_year):
    filtered_df = df[df["year"] == selected_year]

    traces = []
    for continent_name in filtered_df["continent"].unique():
        df_by_continent = filtered_df[filtered_df["continent"] == continent_name]
        traces.append(
            go.Scatter(
                x=df_by_continent["gdpPercap"],
                y=df_by_continent["lifeExp"],
                mode="markers",
                opacity=0.7,
                marker={"size": 15},
                name=continent_name
            )
        )

    layout = go.Layout(
        title="My Plot",
        xaxis={"title": "GDP Per Cap", "type": "log"},
        yaxis={"title": "Life Expectancy"}
    )

    return {
        "data": traces,
        "layout": layout
    }


if __name__ == "__main__":
    app.run()
