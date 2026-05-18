<div align="center">

# 🛒 Retail Demand Forecasting & Inventory Optimization

**Time-series forecasting notebooks** — Predicts retail product demand and estimates inventory risk with TFT and XGBoost models.

<br>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/) [![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-FF6600?style=flat-square)](https://xgboost.readthedocs.io/) [![License](https://img.shields.io/badge/License-Not%20specified-lightgrey?style=flat-square)](https://choosealicense.com/no-permission/)

<br>

[Features](#features) · [Quick Start](#quick-start) · [Usage](#usage) · [Configuration](#configuration) · [Architecture](#architecture) · [AI Models](#ai-models) · [Dependencies](#dependencies) · [License](#license)

</div>

---

## ✨ Features

- **30-day demand forecasting** — Uses store-product time series to predict future demand windows.
- **TFT forecasting workflow** — Trains Temporal Fusion Transformer models with quantile loss, Optuna tuning, and saved checkpoints.
- **XGBoost baseline workflow** — Builds an XGBRegressor pipeline with target encoding, time-series cross-validation, and quantile interval models.
- **Inventory risk analysis** — Calculates stockout rate and estimates safety stock from prediction intervals.
- **Retail feature engineering** — Creates calendar, price ratio, price segment, lag, rolling, and store-level statistical features.
- **Experiment artifacts** — Keeps TensorBoard logs and selected model checkpoints under `lightning_logs/` and `saved_models/`.

---

## 🚀 Quick Start

### Environment setup

```bash
git clone https://github.com/192cm/Retail-Store-Inventory-and-Demand-Forecasting.git
cd Retail-Store-Inventory-and-Demand-Forecasting
conda env create -f environment.yml -n retail-demand
conda activate retail-demand
python -m pip install xgboost notebook
```

### Credentials / config

```bash
python -c "from pathlib import Path; assert Path('sales_data.csv').exists(); print('No API keys required')"
```

### Run

```bash
jupyter notebook
```

---

## 📖 Usage

### Web / GUI

```bash
jupyter notebook
```

| Notebook | Purpose |
|----------|---------|
| `retail-store-inventoty-and-demand-forecasting-using-TFT.ipynb` | Trains and evaluates a Temporal Fusion Transformer with 90-day encoder and 30-day prediction windows |
| `retail-store-inventoty-and-demand-forecasting-Using-XGBoostRegressor.ipynb` | Trains and evaluates XGBoost point and quantile regressors for demand forecasting |

> The repository keeps the original `inventoty` typo in notebook filenames; use the filenames exactly as shown.

### Programmatic / API

```python
import pandas as pd

data = pd.read_csv("sales_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Store ID", "Product ID", "Date"])

stockout_rate = (data["Demand"] > data["Inventory Level"]).mean() * 100
print(f"Stockout Rate: {stockout_rate:.2f}%")
```

---

## ⚙️ Configuration

Everything is controlled through `environment.yml` — no code changes needed to recreate the notebook runtime.

| Key | Default | Description |
|-----|---------|-------------|
| `python` | `3.10.19` | Runtime used by the exported Conda environment |
| `pytorch-forecasting` | `1.6.1` | Builds the TFT `TimeSeriesDataSet` and `TemporalFusionTransformer` |
| `torch` | `2.10.0+cu130` | Runs GPU-accelerated TFT training when CUDA is available |
| `optuna` | `4.7.0` | Tunes TFT and model hyperparameters |
| `scikit-learn` | `1.7.2` | Provides preprocessing, target encoding, pipelines, metrics, and cross-validation |
| `xgboost` | Installed separately | Required by `retail-store-inventoty-and-demand-forecasting-Using-XGBoostRegressor.ipynb` |

---

## 🏗️ Architecture

```
Retail-Store-Inventory-and-Demand-Forecasting/
├── README.md                                                     # project overview
├── environment.yml                                               # Conda runtime export
├── sales_data.csv                                                # retail demand dataset
├── retail-store-inventoty-and-demand-forecasting-using-TFT.ipynb # TFT workflow
├── retail-store-inventoty-and-demand-forecasting-Using-XGBoostRegressor.ipynb # XGBoost workflow
├── lightning_logs/                                               # TensorBoard experiment logs
└── saved_models/                                                 # TFT checkpoint outputs
```

```
sales_data.csv
   │  pandas read_csv
   ▼
EDA
   │  demand trend, feature distributions, stockout rate
   ▼
Feature Engineering
   │  calendar, price, lag, rolling, store-stat features
   ├──────────────▶ XGBoost Pipeline ──▶ point forecast + 95% interval
   │                         │  RMSE, MAPE, safety stock
   │                         ▼
   └──────────────▶ TFT TimeSeriesDataSet ──▶ quantile forecast checkpoint
                             │  TensorBoard logs + model interpretation
                             ▼
                       Forecast Evaluation
```

> The project keeps a tree-based forecasting path and a deep time-series forecasting path side by side so model quality, uncertainty intervals, and inventory signals can be compared.

---

## 🤖 AI Models

### Temporal Fusion Transformer

| Metric | Model | Baseline |
|--------|-------|----------|
| `MAPE` | `10.15%` | `54.79%` |
| `RMSE` | `9.14` | `61.06` |
| `RMSE improvement` | `85.03%` | `0.00%` |

- **Training setup** — Uses `90` encoder steps and `30` prediction steps grouped by `Store ID` and `Product ID`.
- **Uncertainty modeling** — Uses `QuantileLoss` and plots median forecasts with prediction intervals.
- **Best checkpoint** — Loads `saved_models/trial_7/best-checkpoint.ckpt` with hidden size `64`, learning rate `0.008040444263766957`, dropout `0.2433929223767682`, and `2` attention heads.

### XGBoost Regressor

| Metric | Value |
|--------|-------|
| `MAPE` | `16.81%` |
| `RMSE` | `18.35` |
| `RMSE improvement vs Demand_Lag_7` | `66.43%` |
| `Average daily safety stock` | `3,908 units` |

- **Tuning setup** — Uses `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=10)` over `20` candidates.
- **Best parameters** — Uses `n_estimators=1000`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.9`, `reg_alpha=1`, and `reg_lambda=1`.
- **Interval estimation** — Trains separate quantile regressors at `0.025` and `0.975` to estimate a 95% demand range.

---

## 📦 Dependencies

| Dependency | Role |
|------------|------|
| `pandas` | Loads, sorts, groups, and reshapes retail records |
| `numpy` | Computes transformations, metrics, and interval values |
| `matplotlib` | Renders trend, forecast, and interpretation plots |
| `seaborn` | Builds distribution, heatmap, and feature-importance charts |
| `scikit-learn` | Runs preprocessing pipelines, target encoding, metrics, and cross-validation |
| `xgboost` | Trains point and quantile demand regressors |
| `torch` | Runs deep learning training and checkpoint loading |
| `pytorch-forecasting` | Provides TFT, `TimeSeriesDataSet`, baseline model, normalizers, and quantile loss |
| `lightning` | Handles training loops, callbacks, and TensorBoard logging |
| `optuna` | Searches hyperparameters and prunes low-performing trials |

---

## 📄 License

No license file is included in this repository.
