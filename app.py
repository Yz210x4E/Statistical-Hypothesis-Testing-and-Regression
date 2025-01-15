import pandas as pd
import statsmodels.api as sm
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go

# Load datasets
try:
    real_estate_data = pd.read_csv('data/real-estate.csv')
    wine_quality_red_data = pd.read_csv('data/winequality-red.csv', sep=';')
    wine_quality_white_data = pd.read_csv('data/winequality-white.csv', sep=';')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Correlation matrices
real_estate_corr = real_estate_data.corr()
wine_quality_red_corr = wine_quality_red_data.corr()
wine_quality_white_corr = wine_quality_white_data.corr()

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Data Analysis Dashboard"),
    
    html.Div([
        html.H2("Real Estate Data Correlation"),
        dcc.Graph(
            id='real-estate-correlation',
            figure=px.imshow(real_estate_corr, text_auto=True, color_continuous_scale='Viridis')
        )
    ]),
    
    html.Div([
        html.H2("Red Wine Quality Data Correlation"),
        dcc.Graph(
            id='red-wine-correlation',
            figure=px.imshow(wine_quality_red_corr, text_auto=True, color_continuous_scale='Viridis')
        )
    ]),
    
    html.Div([
        html.H2("White Wine Quality Data Correlation"),
        dcc.Graph(
            id='white-wine-correlation',
            figure=px.imshow(wine_quality_white_corr, text_auto=True, color_continuous_scale='Viridis')
        )
    ]),
    
    html.Div([
        html.H2("Linear Regression for Real Estate"),
        dcc.Graph(
            id='real-estate-regression',
            figure=go.Figure()
        )
    ]),
    
    html.Div([
        html.H2("Multiple Regression for Red Wine Quality"),
        dcc.Graph(
            id='red-wine-regression',
            figure=go.Figure()
        )
    ]),
    
    html.Div([
        html.H2("Multiple Regression for White Wine Quality"),
        dcc.Graph(
            id='white-wine-regression',
            figure=go.Figure()
        )
    ])
])

# Regression analysis for Real Estate
@app.callback(
    dash.dependencies.Output('real-estate-regression', 'figure'),
    []
)
def update_real_estate_regression():
    X = real_estate_data['X2 house age']
    y = real_estate_data['Y house price of unit area']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real_estate_data['X2 house age'], y=y, mode='markers', name='Data Points'))
    fig.add_trace(go.Scatter(x=real_estate_data['X2 house age'], y=model.predict(X), mode='lines', name='Regression Line'))
    fig.update_layout(title='Real Estate Price Prediction', xaxis_title='House Age', yaxis_title='Price per Unit Area')
    return fig

# Regression analysis for Red Wine Quality
@app.callback(
    dash.dependencies.Output('red-wine-regression', 'figure'),
    []
)
def update_red_wine_regression():
    X = wine_quality_red_data.drop(columns=['quality'])
    y = wine_quality_red_data['quality']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    fig = go.Figure()
    for col in X.columns[1:]:  # Skip the constant
        fig.add_trace(go.Scatter(x=X[col], y=model.predict(X), mode='lines', name=f'Regression Line for {col}'))
    fig.update_layout(title='Red Wine Quality Prediction', xaxis_title='Features', yaxis_title='Quality')
    return fig

# Regression analysis for White Wine Quality
@app.callback(
    dash.dependencies.Output('white-wine-regression', 'figure'),
    []
)
def update_white_wine_regression():
    X = wine_quality_white_data.drop(columns=['quality'])
    y = wine_quality_white_data['quality']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    fig = go.Figure()
    for col in X.columns[1:]:  # Skip the constant
        fig.add_trace(go.Scatter(x=X[col], y=model.predict(X), mode='lines', name=f'Regression Line for {col}'))
    fig.update_layout(title='White Wine Quality Prediction', xaxis_title='Features', yaxis_title='Quality')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 