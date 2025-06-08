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

## HTML components

- Create an HTML Div
- Multiple items inside the Div?
  - Create a list to hold the components
- Outside of that list can be a style dictionary
  - `style={'property': 'value'}`

## Troubleshoot

- https://stackoverflow.com/questions/69375384/python-dash-core-components-modulenotfound