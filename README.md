# Sales Volume Prediction

This repository contains a data science project aimed at predicting product sales volume by analyzing various features, such as product positioning, promotional effects, and seasonal attributes. The analysis uses a combination of machine learning and deep learning models, including a Random Forest Regressor and a Multi-Layer Perceptron (MLP), to better understand the drivers of sales volume.

## Project Structure

- **`sales_volume_prediction.ipynb`**: Main notebook covering all stages of data wrangling, exploratory data analysis, feature engineering, and predictive modeling.
- **`data/`**: Directory containing the raw and cleaned datasets.
- **`dcn_tuning/`**: Folder containing configuration files for tuning the deep complex network model.
- **`environment.yml`**: Conda environment file for replicating the project environment with all dependencies.

## Project Workflow

### 1. Data Wrangling
   - **Data Loading and Cleaning**: Initial assessment, handling missing values, removing redundancies, and transforming timestamp formats for accurate time-based analysis.
   - **Data Tidiness**: Dropped low-variance columns such as `currency` and `brand`, and standardized column names to snake_case.

### 2. Exploratory Data Analysis (EDA)
   - **Sales Volume Patterns**: Distribution analysis, seasonality check, and statistical tests (Shapiro-Wilk) to examine normality.
   - **Impact of Product Position and Promotion**: Explored sales trends based on product position (Aisle, End-cap, Front of Store) and promotion status, including ANOVA tests.
   - **Price Analysis**: Correlation and group analysis across price ranges with visualizations to examine the effect of price on sales volume.

### 3. Feature Engineering
   - **Numerical Features**: Standard scaling of `price`.
   - **Categorical Encoding**: One-hot encoding for `product_position`, `promotion`, `section`, and `terms` to enhance feature representation.

### 4. Modeling
   - **Random Forest Regressor**: An ensemble learning approach, tuned using `GridSearchCV` to optimize parameters for robust sales volume predictions.
   - **Multi-Layer Perceptron (MLP)**: Built using TensorFlow and Keras to capture complex relationships in the data. Tuned using `EarlyStopping` and additional layers configured in `dcn_tuning/` to enhance model performance.
   - **TabNet Regressor**: Implemented through PyTorch’s TabNet library to provide an explainable deep learning model, suitable for tabular data and fine-tuned for the dataset's non-linear patterns.

### 5. Results and Insights
   - **Limited Impact of Promotions**: Promotions have minimal influence on sales volume across product positions and price ranges.
   - **Insignificant Effects of Product Position**: Sales volume remains consistent across product positions, with a slight increase at Front of Store.
   - **Price Sensitivity**: Sales are generally lower for high-priced items (above $200), though overall correlation between price and sales volume is weak.

## Getting Started

To replicate this analysis:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-volume-prediction.git
   ```
2. Navigate to the repository:
   ```bash
   cd sales-volume-prediction
   ```
3. Set up the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate sales_volume_env
   ```
4. Launch Jupyter Notebook and open `sales_volume_prediction.ipynb`:
   ```bash
   jupyter notebook
   ```

## Dependencies

All dependencies are managed via `environment.yml`, including:

- **Data Manipulation**: `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`
- **Statistical Analysis**: `scipy`, `statsmodels`
- **Machine Learning**: `scikit-learn`
- **Deep Learning**: `tensorflow` for MLP, `torch` and `pytorch-tabnet` for TabNet

### Notes
The `environment.yml` file is the recommended setup for reproducing this analysis, containing both Conda and pip-installed packages for seamless compatibility.
