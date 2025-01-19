import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px

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

# Interactive Visualization for Real Estate
st.header("Interactive Visualization for Real Estate")
feature = st.selectbox("Select Feature for Regression", ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores'])
X = real_estate_data[feature]
y = real_estate_data['Y house price of unit area']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Display key regression statistics
st.subheader("Regression Results")
results_df = pd.DataFrame({
    'Feature': model.params.index,
    'Coefficient': model.params.values,
    'P-Value': model.pvalues.values
})
st.write(results_df)

# Plotting the regression results
plt.figure(figsize=(10, 6))
plt.scatter(real_estate_data[feature], y, label='Data Points')
plt.plot(real_estate_data[feature], model.predict(X), color='red', label='Regression Line')
plt.title(f'Real Estate Price Prediction based on {feature}')
plt.xlabel(feature)
plt.ylabel('Price per Unit Area')
plt.legend()
st.pyplot()

# Multiple Regression for Red Wine Quality
st.header("Multiple Regression for Red Wine Quality")
X_red = wine_quality_red_data.drop(columns=['quality'])  # Independent variables
y_red = wine_quality_red_data['quality']  # Dependent variable
X_red = sm.add_constant(X_red)  # Adding a constant for the intercept
model_red = sm.OLS(y_red, X_red).fit()

# Display key regression statistics for Red Wine
st.subheader("Red Wine Quality Regression Results")
results_red_df = pd.DataFrame({
    'Feature': model_red.params.index,
    'Coefficient': model_red.params.values,
    'P-Value': model_red.pvalues.values
})
st.write(results_red_df)

# Interactive Visualization for Red Wine Quality
st.subheader("Interactive Visualization for Red Wine Quality")
feature_red = st.selectbox("Select Feature for Red Wine Quality Regression", X_red.columns[1:])
plt.figure(figsize=(10, 6))
plt.scatter(wine_quality_red_data[feature_red], y_red, label='Data Points')
plt.plot(wine_quality_red_data[feature_red], model_red.predict(X_red), color='red', label='Regression Line')
plt.title(f'Red Wine Quality Prediction based on {feature_red}')
plt.xlabel(feature_red)
plt.ylabel('Quality')
plt.legend()
st.pyplot()

# Multiple Regression for White Wine Quality
st.header("Multiple Regression for White Wine Quality")
X_white = wine_quality_white_data.drop(columns=['quality'])  # Independent variables
y_white = wine_quality_white_data['quality']  # Dependent variable
X_white = sm.add_constant(X_white)  # Adding a constant for the intercept
model_white = sm.OLS(y_white, X_white).fit()

# Display key regression statistics for White Wine
st.subheader("White Wine Quality Regression Results")
results_white_df = pd.DataFrame({
    'Feature': model_white.params.index,
    'Coefficient': model_white.params.values,
    'P-Value': model_white.pvalues.values
})
st.write(results_white_df)

# Interactive Visualization for White Wine Quality
st.subheader("Interactive Visualization for White Wine Quality")
feature_white = st.selectbox("Select Feature for White Wine Quality Regression", X_white.columns[1:])
plt.figure(figsize=(10, 6))
plt.scatter(wine_quality_white_data[feature_white], y_white, label='Data Points')
plt.plot(wine_quality_white_data[feature_white], model_white.predict(X_white), color='red', label='Regression Line')
plt.title(f'White Wine Quality Prediction based on {feature_white}')
plt.xlabel(feature_white)
plt.ylabel('Quality')
plt.legend()
st.pyplot() 