import dash
from dash import dcc, html

app = dash.Dash()

markdown_text = """
### Dash and Markdown
Dash uses the [CommonMark](http://commonmark.org/) specification
**bold text**
*italics*
inline `code` snippets
"""

app.layout = html.Div(
    [
        dcc.Markdown(
            children=markdown_text
        )
    ]
)

if __name__ == "__main__":
    app.run()