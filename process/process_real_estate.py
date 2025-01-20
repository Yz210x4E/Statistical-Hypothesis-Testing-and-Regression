import pandas as pd
import statsmodels.api as sm
import numpy as np

def load_real_estate_data(file_path):
    """Load real estate data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def analyze_real_estate_data(data):
    """Perform analysis on the real estate data."""
    # Correlation analysis
    correlation_matrix = data.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Multiple linear regression: Predicting house price based on all features
    X = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 
              'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]  # Independent variables
    y = data['Y house price of unit area']  # Dependent variable
    X = sm.add_constant(X)  # Adding a constant for the intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())

if __name__ == "__main__":
    file_path = 'data/real-estate.csv'
    real_estate_data = load_real_estate_data(file_path)
    analyze_real_estate_data(real_estate_data)