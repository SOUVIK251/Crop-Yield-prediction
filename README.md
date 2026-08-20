# Crop Yield Prediction

Multi-year recursive yield prediction model for food grains in West Bengal (Kharif and Rabi seasons).

## Features
- **Data Preprocessing**: Filters and standardizes historical crop yield data for West Bengal.
- **Feature Engineering**: Calculates historical lag values (`lag1`, `lag2`), 3-year rolling mean (`roll3_mean`), and percentage change (`pct_change`).
- **Machine Learning & Forecasting**:
  - Uses `RandomForestRegressor` with recursive forecasting for 2025, 2026, and 2027 yields where sufficient history exists.
  - Fallback mechanisms including linear trend fitting and persistence model for crops with limited data.
- **Evaluation**: Calculates Test MAE and R² metrics on held-out evaluation splits.

## Getting Started

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage
Place the input data CSV file `State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv` in the root folder, then run:
```bash
python main.py
```
The predicted yields will be exported to `wb_yield_multiyear_recursive.csv`.
