import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Configure matplotlib for better visualization
sns.set_style("whitegrid")  # Use Seaborn's styling

# Function to reset matplotlib figure
def clear_figure():
    plt.clf()
    plt.close('all')

try:
    # Load datasets
    @st.cache_data  # Cache the data loading
    def load_data():
        real_estate_data = pd.read_csv('data/real-estate.csv')
        wine_quality_red_data = pd.read_csv('data/winequality-red.csv', sep=';')
        wine_quality_white_data = pd.read_csv('data/winequality-white.csv', sep=';')
        return real_estate_data, wine_quality_red_data, wine_quality_white_data

    real_estate_data, wine_quality_red_data, wine_quality_white_data = load_data()

    # Title and description
    st.title("Data Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes three datasets:
    * Real Estate Pricing
    * Red Wine Quality
    * White Wine Quality
    """)

    st.sidebar.header("Contents")
st.sidebar.markdown("""
- [Home](#home)
- [Real Estate Analysis](#real-estate-analysis)
- [Wine Quality Analysis](#wine-quality-analysis)
""")
    st.sidebar.header("Team Members")
    st.sidebar.markdown("""
    - **Youssef A. Zahran**
    - **Dionis Leka**
    - ****
    """)

    # Create three columns for correlation matrices
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Real Estate Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(real_estate_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Real Estate Correlation Matrix")
        st.pyplot(fig)
        clear_figure()

    with col2:
        st.header("Red Wine Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(wine_quality_red_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("Red Wine Correlation Matrix")
        st.pyplot(fig)
        clear_figure()

    with col3:
        st.header("White Wine Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(wine_quality_white_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title("White Wine Correlation Matrix")
        st.pyplot(fig)
        clear_figure()

    # Real Estate Analysis
    st.header("Real Estate Analysis")
    feature = st.selectbox(
        "Select Feature for Real Estate Analysis",
        ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']
    )

    # Perform regression
    X = sm.add_constant(real_estate_data[feature])
    y = real_estate_data['Y house price of unit area']
    model = sm.OLS(y, X).fit()

    # Display regression results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regression Statistics")
        results_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'P-value (F-stat)'],
            'Value': [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue]
        })
        st.table(results_df)

    with col2:
        st.subheader("Coefficient Analysis")
        coef_df = pd.DataFrame({
            'Feature': ['Intercept', feature],
            'Coefficient': model.params,
            'P-Value': model.pvalues
        })
        st.table(coef_df)

    # Plot regression
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(real_estate_data[feature], y, alpha=0.5)
    ax.plot(real_estate_data[feature], model.predict(X), color='red', linewidth=2)
    ax.set_xlabel(feature)
    ax.set_ylabel('House Price per Unit Area')
    ax.set_title(f'Real Estate Price vs {feature}')
    st.pyplot(fig)
    clear_figure()

    # Wine Quality Analysis
    st.header("Wine Quality Analysis")
    wine_type = st.radio("Select Wine Type", ["Red Wine", "White Wine"])

    if wine_type == "Red Wine":
        wine_data = wine_quality_red_data
    else:
        wine_data = wine_quality_white_data

    # Feature selection for wine analysis
    wine_feature = st.selectbox(
        "Select Feature for Wine Analysis",
        wine_data.drop('quality', axis=1).columns
    )

    # Perform wine regression
    X_wine = sm.add_constant(wine_data[wine_feature])
    y_wine = wine_data['quality']
    wine_model = sm.OLS(y_wine, X_wine).fit()

    # Display wine regression results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wine Regression Statistics")
        wine_results_df = pd.DataFrame({
            'Metric': ['R-squared', 'Adj. R-squared', 'F-statistic', 'P-value (F-stat)'],
            'Value': [wine_model.rsquared, wine_model.rsquared_adj, wine_model.fvalue, wine_model.f_pvalue]
        })
        st.table(wine_results_df)

    with col2:
        st.subheader("Wine Coefficient Analysis")
        wine_coef_df = pd.DataFrame({
            'Feature': ['Intercept', wine_feature],
            'Coefficient': wine_model.params,
            'P-Value': wine_model.pvalues
        })
        st.table(wine_coef_df)

    # Plot wine regression
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(wine_data[wine_feature], y_wine, alpha=0.5)
    ax.plot(wine_data[wine_feature], wine_model.predict(X_wine), color='red', linewidth=2)
    ax.set_xlabel(wine_feature)
    ax.set_ylabel('Wine Quality')
    ax.set_title(f'{wine_type} Quality vs {wine_feature}')
    st.pyplot(fig)
    clear_figure()

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please make sure all required data files are present in the correct location.")
