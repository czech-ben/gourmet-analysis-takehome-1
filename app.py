import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input

from sklearn import linear_model

data = pd.read_csv("data/daily_data.csv")
data.sort_values("date", inplace=True)

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Company Order Data Analysis"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.H1(
                    children="Company Order Data", className="header-title"
                ),
                html.P(
                    children="Analyze the behavior of ecommerce order data over time"
                    " for a group of companies",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Company", className="menu-title"),
                        dcc.Dropdown(
                            id="company-filter",
                            options=[
                                {"label": company, "value": company}
                                for company in np.sort(data.company_id.unique())
                            ],
                            value="Jim's Gym Supplies",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="Date Range",
                            className="menu-title"
                            ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=pd.to_datetime(data.date).min().date(),
                            max_date_allowed=pd.to_datetime(data.date).max().date(),
                            start_date=pd.to_datetime(data.date).min().date(),
                            end_date=pd.to_datetime(data.date).max().date(),
                        ),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="revenue-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="monthly-revenue-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="new-customers-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="repeat-customer-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="churn-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="ltv-chart", config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    [Output("revenue-chart", "figure"),
     Output("new-customers-chart", "figure"),
     Output("monthly-revenue-chart", "figure"),
     Output("repeat-customer-chart", "figure"),
     Output("churn-chart", "figure"),
     Output("ltv-chart", "figure")],
    [
        Input("company-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_charts(company_id, start_date, end_date):
    mask = (
        (data.company_id == company_id)
        & (data.date >= start_date)
        & (data.date <= end_date)
    )
    filtered_data = data.loc[mask, :]

    price_chart_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["revenue"].cumsum(),
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Cumulative Revenue",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 10, "title": "Date"},
            "yaxis": {"tickprefix": "$", "fixedrange": True, "title": "Dollars"},
            "colorway": ["#636EFA"],
        },
    }

    new_customers_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["new_customers"].cumsum(),
                "type": "lines",
                "hovertemplate": "%{y:.0f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Unique Customers",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 10, "title": "Date"},
            "yaxis": {"fixedrange": True, "title": "Customers"},
            "colorway": ["#EF553B"],
        },
    }

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
              'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_revenues = []
    for month in months:
        column_name = "month_" + month
        revenue = filtered_data[filtered_data[column_name] == 1].revenue.sum()
        monthly_revenues.append(revenue)


    monthly_revenue = {
        "data": [
             {'x': months, 'y': monthly_revenues, 'type': 'bar', 'name': 'Month'},
        ],
        "layout": {
            "title": {
                "text": "Monthly Revenue",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 12, "title": "Date"},
            "yaxis": {"tickprefix": "$", "fixedrange": True, "title": "Dollars"},
            "colorway": ["#OOCC96"],
        },
    }

    repeat_customer_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["repeat_customer_rate"],
                "type": "lines",
                "hovertemplate": "%{y:,.0%}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Repeat Customer Rate",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 10, "title": "Date"},
            "yaxis": {"fixedrange": True, "title": "Percentage", "tickformat": ',.0%',},
            "colorway": ["#AB63FA"],
        },
    }

    churn_rate_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["churn_rate"],
                "type": "lines",
                "hovertemplate": "$%{y:,.0%}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Ecommerce Churn Rate",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 10, "title": "Date"},
            "yaxis": {"tickformat": ',.0%', "fixedrange": True, "title": "Percentage"},
            "colorway": ["#FFA15A"],
        },
    }

    customer_ltv_figure = {
        "data": [
            {
                "x": filtered_data["date"],
                "y": filtered_data["customer_lifetime_value"],
                "type": "lines",
                "hovertemplate": "$%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Customer Lifetime Value",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True, "nticks": 10, "title": "Date"},
            "yaxis": {"fixedrange": True, "title": "Dollars", "tickprefix": "$"},
            "colorway": ["#19D3F3"],
        },
    }

    return price_chart_figure, new_customers_figure, monthly_revenue, repeat_customer_figure, churn_rate_figure, customer_ltv_figure


if __name__ == "__main__":
    app.run_server(debug=True)