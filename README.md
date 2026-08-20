# West Bengal Crop Yield Prediction (2026–2027)

A machine-learning project for predicting crop yield in West Bengal for the years 2026 and 2027 using Random Forest Regression and historical agricultural yield data.

## Project Overview

This project uses historical crop-yield records to build Crop × Season specific Random Forest regression models. The workflow includes data cleaning, West Bengal filtering, temporal feature engineering, model training, evaluation, recursive forecasting, and visualization.

The model predicts future crop yield in Kg/Ha (kilograms per hectare).

## Objectives

- Clean and prepare the historical agricultural dataset.
- Extract records belonging to West Bengal.
- Organize the data by Crop Name and Season.
- Generate historical time-dependent features.
- Train Random Forest Regression models.
- Evaluate the models using MAE and R².
- Predict crop yield for 2026 and 2027.
- Generate visual representations of the predictions.
- Save the final prediction results as a CSV file.

## Dataset

**Dataset**: State-wise Yield of Food Grains in Kharif and Rabi Seasons

**Expected dataset file**:
`State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv`

### Important fields used by the project:

| Field | Description |
| --- | --- |
| Year | Agricultural year |
| State / Uts | State or Union Territory |
| Crop Name | Name of crop/crop group |
| Season | Agricultural season |
| Yield | Crop yield in Kg/Ha |

Place the original dataset file in the project directory before running the code.

## Methodology

```
Historical Agricultural Dataset
            ↓
Data Loading
            ↓
Year Extraction and Cleaning
            ↓
West Bengal Filtering
            ↓
Crop × Season Grouping
            ↓
Feature Engineering
            ↓
Train/Test Evaluation
            ↓
Random Forest Regression
            ↓
Final Model Training
            ↓
Recursive Forecasting
            ↓
2026 Prediction
            ↓
2027 Prediction
            ↓
CSV Output + Visualization
```

## Feature Engineering

Five temporal features are generated for each Crop × Season group:
- `year_idx` — sequential position of the observation.
- `lag1` — previous yield value.
- `lag2` — yield value from two observations earlier.
- `roll3_mean` — rolling mean based on recent yield observations.
- `pct_change` — relative change from the previous yield value.

## Random Forest Model

The implementation uses:
```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
```

### Main Configuration

| Parameter | Value |
| --- | --- |
| Model | RandomForestRegressor |
| Number of Trees | 300 |
| Random State | 42 |
| Task | Regression |
| Target | Crop Yield (Kg/Ha) |

## Model Evaluation

For groups with sufficient historical observations, the model is evaluated using a train/test split.
- **Test size**: 25%
- **Random state**: 42
- **Metrics**:
  - Mean Absolute Error (MAE)
  - R² Score

## Recursive Forecasting

The model uses recursive forecasting for future years:

```
Historical Yield Data
        ↓
Generate Features
        ↓
Predict Future Year
        ↓
Add Prediction to History
        ↓
Generate Updated Features
        ↓
Predict Next Future Year
```

The final target years are 2026 and 2027.

## Handling Insufficient Data

Some Crop × Season groups may not contain enough historical observations for reliable Random Forest training.

The implementation therefore contains fallback logic:
- **Linear Trend** when the implemented conditions permit it.
- **Persistence Fallback** when the historical data are too sparse or unsuitable for trend-based forecasting.

The prediction method used for each group is recorded in the output.

## Prediction Output

The final prediction results are saved as:
`wb_yield_multiyear_recursive.csv`

The output includes:
- Crop Name
- Season
- Predicted Yield for 2026
- Predicted Yield for 2027
- Prediction Method
- Test MAE
- Test R²

## Visualization

The project includes visualization code for comparing predicted crop yields for 2026 and 2027.

Generated graphs can be stored in:
`results/`

## Project Structure

```
West-Bengal-Crop-Yield-Prediction/
│
├── data/
│   └── State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv
│
├── src/
│   ├── crop_yield_random_forest.py
│   └── visualization.py
│
├── results/
│   ├── wb_yield_multiyear_recursive.csv
│   └── prediction_graph.png
│
├── report/
│   └── Research_Report.pdf
│
├── README.md
└── requirements.txt
```

Adjust the filenames and folders to match your actual GitHub repository.

## Requirements

Install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
Or:
```bash
pip install -r requirements.txt
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/SOUVIK251/Crop-Yield-prediction.git
   cd Crop-Yield-prediction
   ```

2. **Add the dataset**
   Place the dataset inside the configured `data/` folder.

3. **Run the prediction code**
   ```bash
   python src/crop_yield_random_forest.py
   ```

4. **Run the visualization code**
   ```bash
   python src/visualization.py
   ```

5. **Check the results**
   The prediction file will be generated as:
   `wb_yield_multiyear_recursive.csv`

## Important Results

The supplied experiment contains 31 Crop × Season combinations.

Some reported evaluation results are:

| Crop | Season | MAE | R² |
| --- | --- | --- | --- |
| Oilseeds | Kharif | 31.9 | 0.852 |
| Rice | Total | 64.0 | 0.724 |
| Foodgrains | Total | 60.9 | 0.678 |
| Rice | Kharif | 61.2 | 0.619 |
| Groundnut | Rabi | 165.2 | 0.585 |

The strongest reported R² value is **0.852** for **Oilseeds — Kharif**.

## File Links

- **Dataset File**: [State-wise Yield of Food Grains in Kharif and Rabi seasons (1).xlsx](https://1drv.ms/x/c/F78F2EA9E20C8C53/AaZWKxHTW71PgnccmR-FGKc?e=5f4HIh)

## Limitations

- The current model mainly uses historical yield and derived temporal features.
- Rainfall and temperature are not included.
- Soil properties are not included.
- Fertilizer and irrigation information are not included.
- Some Crop × Season groups have limited historical observations.
- Recursive forecasting can propagate prediction errors.
- Model performance varies across Crop × Season groups.
- Some groups use fallback forecasting because of insufficient data.
- The evaluation uses a random train/test split rather than strict chronological validation.

## Future Scope

- Integrate rainfall, temperature, humidity, and other weather variables.
- Add soil, fertilizer, and irrigation information.
- Use chronological or walk-forward validation.
- Compare Random Forest with XGBoost, Gradient Boosting, Extra Trees, and ANN.
- Perform hyperparameter optimization.
- Analyze feature importance.
- Generate prediction intervals.
- Extend prediction beyond 2027.
- Develop an interactive crop-yield prediction interface.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Random Forest Regression

## Author

**Souvik Kundu**  
Electronics and Communication Engineering (ECE)  
IEEE SMC Student Branch Chapter, KGEC  
Research Internship Programme 2026
