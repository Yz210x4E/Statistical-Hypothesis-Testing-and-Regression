import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load datasets
real_estate_data = pd.read_csv('data/real-estate.csv')
wine_quality_red_data = pd.read_csv('data/winequality-red.csv', sep=';')
wine_quality_white_data = pd.read_csv('data/winequality-white.csv', sep=';')

# Title
st.title("Data Analysis Dashboard")

# Correlation matrices
st.header("Real Estate Data Correlation")
real_estate_corr = real_estate_data.corr()
sns.heatmap(real_estate_corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot()

st.header("Red Wine Quality Data Correlation")
wine_quality_red_corr = wine_quality_red_data.corr()
sns.heatmap(wine_quality_red_corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot()

st.header("White Wine Quality Data Correlation")
wine_quality_white_corr = wine_quality_white_data.corr()
sns.heatmap(wine_quality_white_corr, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot()

# Linear Regression for Real Estate
st.header("Linear Regression for Real Estate")
X = real_estate_data['X2 house age']
y = real_estate_data['Y house price of unit area']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Plotting the regression results
plt.figure(figsize=(10, 6))
plt.scatter(real_estate_data['X2 house age'], y, label='Data Points')
plt.plot(real_estate_data['X2 house age'], model.predict(X), color='red', label='Regression Line')
plt.title('Real Estate Price Prediction')
plt.xlabel('House Age')
plt.ylabel('Price per Unit Area')
plt.legend()
st.pyplot()

# Multiple Regression for Red Wine Quality
st.header("Multiple Regression for Red Wine Quality")
X_red = wine_quality_red_data.drop(columns=['quality'])  # Independent variables
y_red = wine_quality_red_data['quality']  # Dependent variable
X_red = sm.add_constant(X_red)  # Adding a constant for the intercept
model_red = sm.OLS(y_red, X_red).fit()

# Display regression summary
st.subheader("Red Wine Quality Regression Summary")
st.write(model_red.summary())

# Plotting the regression results for Red Wine Quality
st.subheader("Red Wine Quality Predictions")
plt.figure(figsize=(10, 6))
for col in X_red.columns[1:]:  # Skip the constant
    plt.scatter(X_red[col], model_red.predict(X_red), label=f'Predicted Quality based on {col}')
plt.title('Red Wine Quality Prediction')
plt.xlabel('Features')
plt.ylabel('Predicted Quality')
plt.legend()
st.pyplot()

# Multiple Regression for White Wine Quality
st.header("Multiple Regression for White Wine Quality")
X_white = wine_quality_white_data.drop(columns=['quality'])  # Independent variables
y_white = wine_quality_white_data['quality']  # Dependent variable
X_white = sm.add_constant(X_white)  # Adding a constant for the intercept
model_white = sm.OLS(y_white, X_white).fit()

# Display regression summary
st.subheader("White Wine Quality Regression Summary")
st.write(model_white.summary())

# Plotting the regression results for White Wine Quality
st.subheader("White Wine Quality Predictions")
plt.figure(figsize=(10, 6))
for col in X_white.columns[1:]:  # Skip the constant
    plt.scatter(X_white[col], model_white.predict(X_white), label=f'Predicted Quality based on {col}')
plt.title('White Wine Quality Prediction')
plt.xlabel('Features')
plt.ylabel('Predicted Quality')
plt.legend()
st.pyplot() 