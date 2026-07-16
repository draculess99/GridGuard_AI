import pandas as pd
from backend.data import generate_synthetic_demand

def main():
    print("Generating 2 years of synthetic hourly data for Kaggle simulation...")
    df = generate_synthetic_demand(days=730, seed=42)
    
    df = df.rename(columns={"timestamp": "Datetime", "demand_mw": "PJME_MW"})
    df = df[["Datetime", "PJME_MW"]]
    df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    out_path = "data/kaggle/hourly_energy_consumption.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
