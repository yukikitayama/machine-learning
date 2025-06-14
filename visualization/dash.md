# Dash

- Instead of creating an .html file, Dash will profuce a dashboard web application at a URL
- You can deploy your dashboards online.
- Dash apps are composed of two parts
  - Layout of the app
    - Describes what the application looks like
  - Interactivity of the application
- Dash offers two distinct component libraries
  - `dash_html_components`
    - Library which has component for Every HTML tag
    - Describes the layout of the page
  - `dash_core_components`
    - Higher-level interactive components that are generated with JavaScript, HTML, CSS through React.js library
    - Describes the individual graphs themselves
- Dash creates a Flask application by `dash.Dash()`
- Technically, no knowledge of HTML or CSS is needed to create a Dash Dashboard, but to stylize and customize Dash Dashboards, knowledge of HTML and CSS will help out a lot!

## Coding tips

- Put each dropdown menu in a separate html.Div, so that it's easy to maintain styling.

## Image

```python
import base64

def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, "rb").read())
    return f"data:image/png;base64,{encoded.decode()}"
```

## HTML components

- Create an HTML Div
- Multiple items inside the Div?
  - Create a list to hold the components
- Outside of that list can be a style dictionary
  - `style={'property': 'value'}`

## Dash callback

- Callback is for interaction
- Steps
  - Create a function to return some desired output
  - Decorate that function with an `@app.callback` decorator
    - Set an `Output` to a component id
    - Set an `Input` to a component id
  - Connect the desired properties

## Markdown text

- Markdown text allows for links, italics, bold text, bullet lists, etc.

## Troubleshoot

- https://stackoverflow.com/questions/69375384/python-dash-core-components-modulenotfound
- `help(html.Div)`