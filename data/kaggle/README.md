# Kaggle historical data directory

GridGuard does not redistribute a Kaggle competition or dataset file. Download the hourly energy dataset under its applicable Kaggle terms, then either:

1. upload the CSV or ZIP through the Streamlit sidebar; or
2. place a CSV at `data/kaggle/hourly_energy_consumption.csv`; or
3. set `GRIDGUARD_KAGGLE_DATA_PATH` to a different local CSV/ZIP path.

The adapter detects common PJM-style structures, including:

```text
Datetime,PJME_MW
2002-01-01 01:00:00,30393
```

A ZIP containing several regional CSV files is also supported. The app lets the operator select the CSV, timestamp column, demand series, optional temperature column, source timezone, region label, and missing-hour policy.

Required logical fields:

- hourly timestamp;
- electricity demand/load in MW.

Optional fields:

- temperature in Fahrenheit;
- holiday indicator.

When temperature is unavailable, GridGuard uses a clearly identified seasonal proxy rather than claiming observed weather.
