# Sales Volume Prediction

This repository contains a comprehensive data science project focused on predicting sales volume based on various product features, product positioning, and promotional effects. The project implements exploratory data analysis (EDA), data cleaning, and multiple machine learning models to understand the drivers behind sales volume.

## Project Structure

- `sales_volume_prediction.ipynb`: Main notebook covering all stages of the analysis, from data cleaning to predictive modeling.
- `data/`: Directory containing the raw and processed data files.
- `dcn_tuning/`: Folder for configurations related to deep complex network tuning.

## Project Workflow

### 1. Data Cleaning and Preparation
- **Data Loading and Inspection**: Initial assessment of data structure and identifying missing or redundant columns.
- **Data Quality Assessment**: Cleaning steps include handling missing values, removing duplicates, converting timestamps, and standardizing column names.
- **Data Transformation**: Conversion of specific columns like `scraped_at` to datetime format for temporal analysis and dropping low-variance columns such as `currency` and `brand` to streamline the dataset.

### 2. Exploratory Data Analysis (EDA)
- **Sales Volume Distribution**: Visualization and statistical tests confirm a balanced distribution of sales volume, with slight non-normality.
- **Product Position and Promotion Impact**: Analysis using boxplots and ANOVA tests shows minimal influence of position or promotions on sales volume.
- **Seasonality**: Seasonal products do not show significant variation in sales volume compared to non-seasonal products.
- **Price Impact**: Slightly lower sales volume for products priced over $200, though price has a weak overall correlation with sales volume.

### 3. Feature Engineering
- **Numerical Transformation**: Price scaling using `StandardScaler`.
- **Categorical Encoding**: One-hot encoding for columns like `product_position`, `promotion`, `section`, and `terms` to incorporate categorical variables effectively.

### 4. Modeling
   - **Random Forest Regressor**: Initial model for robust prediction, tuned via `GridSearchCV` for optimal hyperparameters.
   - **Deep Complex Network (DCN)**: Tuned with configurations in `dcn_tuning/` for enhanced non-linear modeling.

## Key Insights
- **Sales Volume Drivers**: Position, promotion, seasonality, and price have limited statistical significance on sales volume in this dataset.
- **Promotions and Seasonality**: Limited effectiveness of promotions and seasonality on increasing sales volume.
- **Product Position**: Slight increase in sales at the Front of Store, but not statistically significant.

## Results
The project provides interpretable insights into factors affecting sales volume, concluding with a machine learning model that moderately predicts sales volume based on the dataset’s features.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-volume-prediction.git
   ```
2. Navigate to the repository:
   ```bash
   cd sales-volume-prediction
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook to reproduce the analysis and predictions.

## Dependencies
The project requires:
- `numpy`, `pandas` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for machine learning models
- `tensorflow`, `pytorch_tabnet` for deep learning models