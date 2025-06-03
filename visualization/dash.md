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
  - `dash_core_components`
    - Higher-level interactive components that are generated with JavaScript, HTML, CSS through React.js library
- Dash creates a Flask application by `dash.Dash()`

## Troubleshoot

- https://stackoverflow.com/questions/69375384/python-dash-core-components-modulenotfound