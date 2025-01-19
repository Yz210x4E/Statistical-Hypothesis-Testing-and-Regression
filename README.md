# Statistical Hypothesis Testing and Regression Analysis Dashboard

This interactive web application provides powerful tools for statistical analysis and visualization of three key datasets: Real Estate Pricing, Red Wine Quality, and White Wine Quality. Built with Streamlit, the dashboard enables users to perform in-depth statistical hypothesis testing and regression analysis through an intuitive interface.

📊 **[Try the Live Dashboard](https://statistical-hypothesis-testing-and-regrysis-jkwgpk9btxfeqw4a59.streamlit.app/)**

## Project Overview

The dashboard serves as a comprehensive analytical tool that helps users understand complex relationships within datasets through:

- Statistical hypothesis testing to validate relationships between variables
- Interactive regression analysis with customizable parameters
- Dynamic visualization of correlations and relationships
- User-friendly interface with intuitive navigation

## Ways to Access

You can use this dashboard in two ways:

1. **Live Web Application**: Access the deployed version instantly through your browser:
   - Visit [https://statistical-hypothesis-testing-and-regrysis-jkwgpk9btxfeqw4a59.streamlit.app/](https://statistical-hypothesis-testing-and-regrysis-jkwgpk9btxfeqw4a59.streamlit.app/)
   - No installation required
   - Start analyzing data immediately

2. **Local Installation**: Run the application on your local machine for development or offline use by following the installation instructions below.

## Features

### Real Estate Analysis
The real estate analysis module helps users understand property pricing factors through:

- Correlation matrices visualizing relationships between all features in the dataset
- Linear regression analysis examining how factors like house age, MRT station proximity, and convenience store density affect prices
- Interactive scatter plots with regression lines for visual analysis of feature relationships
- Detailed statistical outputs including p-values, R-squared values, and confidence intervals

### Wine Quality Analysis
Both red and white wine datasets can be analyzed to understand quality determinants:

- Comprehensive correlation analysis of wine characteristics
- Regression modeling to identify key quality factors like alcohol content, acidity, and pH
- Interactive visualizations showing relationships between chemical properties and wine quality
- Comparative analysis between red and white wine characteristics

### User Interface
The application features a carefully designed interface for optimal user experience:

- Intuitive sidebar navigation for seamless movement between analysis modules
- Interactive widgets allowing users to customize their analysis parameters
- Clear visualization controls for adjusting plot parameters
- Detailed explanations of statistical concepts and interpretations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Yz210x4E/Statistical-Hypothesis-Testing-and-Regrysis.git
cd Statistical-Hypothesis-Testing-and-Regrysis
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run app_streamlit.py
```

## Dataset Information

The application analyzes three distinct datasets from the UCI Machine Learning Repository:

1. Real Estate Pricing Dataset
   - Features include house age, distance to MRT stations, convenience store count
   - Target variable: House price per unit area
   - Used for property value prediction and feature importance analysis

2. Red Wine Quality Dataset
   - Contains physicochemical properties of red wines
   - Features include alcohol content, acidity levels, pH, and other chemical properties
   - Target variable: Wine quality score

3. White Wine Quality Dataset
   - Similar structure to red wine dataset but for white wines
   - Enables comparative analysis between red and white wine characteristics
   - Facilitates understanding of quality determinants across wine types

## Technologies

The project leverages several powerful Python libraries:

- Streamlit: Powers the web interface and interactive components
- Pandas: Handles data manipulation and analysis
- StatsModels: Provides statistical modeling capabilities
- Seaborn & Matplotlib: Generate static visualizations
- Plotly: Creates interactive plots
- NumPy: Supports numerical computations

## Contributing

We welcome contributions to improve the dashboard. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions, feedback, or collaboration opportunities:

- Youssef A. Zahran - youssef.zzzz802@gmail.com
- Dionis Leka - dleka22@epoka.edu.al
- Xhesika Gjikola - xgjikola21@epoka.edu.al

## Acknowledgments

- Epoka University for providing institutional support
- UCI Machine Learning Repository for the datasets
- The Streamlit team for their excellent framework

---

Developed with ❤️ at Epoka University
