import pandas as pd
import statsmodels.api as sm

def load_wine_quality_data(file_path):
    """Load wine quality data from a CSV file."""
    data = pd.read_csv(file_path, sep=';')  # Using semicolon as separator
    return data

def analyze_wine_quality_data(data):
    """Perform analysis on the wine quality data."""
    # Correlation analysis
    correlation_matrix = data.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Multiple regression: Predicting quality based on other features
    X = data.drop(columns=['quality'])  # Independent variables
    y = data['quality']  # Dependent variable
    X = sm.add_constant(X)  # Adding a constant for the intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())

if __name__ == "__main__":
    file_path = 'data/winequality-red.csv'
    wine_quality_data = load_wine_quality_data(file_path)
    analyze_wine_quality_data(wine_quality_data) 