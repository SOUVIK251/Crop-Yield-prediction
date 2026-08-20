import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re
import os
import warnings
warnings.filterwarnings("ignore")

# 1. Setup and Data Loading
INPUT_FILE = "data/State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv" if os.path.exists("data/State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv") else "State-wise Yield of Food Grains in Kharif and Rabi seasons (4).csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wb_predicted_yield_2026-2027.csv")

def run_prediction():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Please place the CSV file in the working directory.")
        return

    YIELD_COL = "Yield (UOM:Kg/Ha(KilogramperHectare)), Scaling Factor:1"
    df["YearNum"] = df["Year"].apply(lambda x: int(re.search(r"(\d{4})", str(x)).group(1)) if re.search(r"(\d{4})", str(x)) else np.nan)
    df.dropna(subset=["YearNum"], inplace=True) 
    df["YearNum"] = df["YearNum"].astype(int)
    wb = df[df["State / Uts"].str.contains("West Bengal", case=False, na=False)].copy()
    wb = wb.dropna(subset=[YIELD_COL])
    wb = wb[["Crop Name", "Season", "YearNum", YIELD_COL]].rename(columns={YIELD_COL: "Yield"})

    def build_features(group):
        g = group.sort_values("YearNum").reset_index(drop=True)
        g["year_idx"] = np.arange(len(g))
        g["lag1"] = g["Yield"].shift(1)
        g["lag2"] = g["Yield"].shift(2)
        g["roll3_mean"] = g["Yield"].rolling(3, min_periods=1).mean().shift(1)
        g["pct_change"] = g["Yield"].pct_change().replace([np.inf, -np.inf], 0)
        return g

    FEATURES = ["year_idx", "lag1", "lag2", "roll3_mean", "pct_change"]
    results = []

    # 2. Train + Forecast per group
    for (crop, season), group in wb.groupby(["Crop Name", "Season"]):
        g = build_features(group)
        last_year = g["YearNum"].max()
        if len(g) < 6:
            gap_to_2027 = 2027 - last_year
            if len(g) >= 4 and gap_to_2027 <= 6:
                x = g["YearNum"].values
                y = g["Yield"].values
                coeffs = np.polyfit(x, y, 1)
                pred_2025 = np.polyval(coeffs, 2025)
                pred_2026 = np.polyval(coeffs, 2026)
                pred_2027 = np.polyval(coeffs, 2027)
                method = "Linear trend (insufficient data for RF)"
            else:
                pred_2025 = pred_2026 = pred_2027 = g["Yield"].iloc[-1]
                method = "Persistence Fallback"
            results.append({
                "Crop Name": crop, "Season": season, "Last_Available_Year": 2025,
                "Last_Available_Yield (2025 Pred)": round(max(pred_2025, 0), 1),
                "Predicted_Yield_2026": round(max(pred_2026, 0), 1),
                "Predicted_Yield_2027": round(max(pred_2027, 0), 1),
                "Method": method, "Test_MAE": np.nan, "Test_R2": np.nan
            })
            continue
        g_feat = g.dropna(subset=FEATURES)
        X = g_feat[FEATURES]
        y = g_feat["Yield"]

        # 3. Split for metrics
        if len(X) >= 8:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            eval_model = RandomForestRegressor(n_estimators=300, random_state=42)
            eval_model.fit(X_train, y_train)
            test_preds = eval_model.predict(X_test)
            mae = mean_absolute_error(y_test, test_preds)
            r2 = r2_score(y_test, test_preds) if len(y_test) > 1 else np.nan
        else:
            mae, r2 = np.nan, np.nan

        # 4. Final model on full history
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        history = g.copy()
        preds_map = {}
        for yr in [2025, 2026, 2027]:
            recent = history["Yield"].values
            x_next = pd.DataFrame([{
                "year_idx": history["year_idx"].iloc[-1] + 1,
                "lag1": recent[-1],
                "lag2": recent[-2] if len(recent) >= 2 else recent[-1],
                "roll3_mean": np.mean(recent[-3:]),
                "pct_change": (recent[-1] - recent[-2])/recent[-2] if len(recent) >= 2 and recent[-2] != 0 else 0
            }])
            next_pred = model.predict(x_next)[0]
            preds_map[yr] = next_pred
            history = pd.concat([history, pd.DataFrame([{"YearNum": yr, "Yield": next_pred, "year_idx": x_next["year_idx"].iloc[0]}])], ignore_index=True)
        results.append({
            "Crop Name": crop, "Season": season, "Last_Available_Year": 2025,
            "Last_Available_Yield (2025 Pred)": round(preds_map[2025], 1),
            "Predicted_Yield_2026": round(preds_map[2026], 1),
            "Predicted_Yield_2027": round(preds_map[2027], 1),
            "Method": "Random Forest (recursive forecast)",
            "Test_MAE": round(mae, 1) if not np.isnan(mae) else np.nan,
            "Test_R2": round(r2, 3) if not np.isnan(r2) else np.nan
        })
    results_df = pd.DataFrame(results).sort_values(["Crop Name", "Season"])
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    try:
        display(results_df)
    except NameError:
        print(results_df.to_string())
        
    print(f"\nTrained and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_prediction()
