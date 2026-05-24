# Mobile Traffic Forecasting Pipeline

This repository contains the complete spatiotemporal machine learning pipeline for analyzing and forecasting mobile network traffic in the city of Milan.

The project processes an extracted **27.3 GB raw spatiotemporal text dataset** from Telecom Italia Mobile (TIM) Milan Grid (composed of 62 uncompressed daily logs), performs deep exploratory data analysis, trains and evaluates three sequence models (Holt-Winters Exponential Smoothing, LSTM, and GRU) for one-step-ahead forecasting.


---

## Dataset

> **The `Dataset/` folder is not included in this repository** due to its large size (~27.3 GB uncompressed). You must download it manually before running the pipeline.

This project uses two datasets from Harvard Dataverse:

- **[Milan Grid Telco Activity (Nov–Dec 2013)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)** — 62 daily `.txt` logs of internet, call, and SMS activity across a 100×100 grid of Milan.
- **[Milan Grid GeoJSON](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QJWLFU)** — GeoJSON file defining the geographic boundaries of the 10,000 grid cells.

### Setup Instructions

1. Download all files from both links above.
2. Create a `Dataset/` folder in the project root if it doesn't already exist.
3. Unzip the daily `.txt` files and place them into the matching subdirectories:

```
Dataset/
├── dataverse_files
├── dataverse_files (1)/    ← files from Nov 1 – Nov 20
├── dataverse_files (2)/    ← files from Nov 21 – Nov 30
├── dataverse_files (3)/    ← files from Dec 1 – Dec 10
├── dataverse_files (4)/    ← files from Dec 11 – Dec 20
├── dataverse_files (5)/    ← files from Dec 21 – Dec 30
├── dataverse_files (6)/    ← files from Dec 31 – Jan 1
└── milano-grid.geojson     ← place directly in Dataset/
```

4. Once the dataset is in place, follow the installation steps below and run `python3 main.py`.

---

## Getting Started

### Prerequisites

- **Python 3.10–3.13**
- **macOS** (tested on Apple Silicon M1 Pro; also works on Intel Mac and Linux)
- **~25 GB free disk space** for the raw dataset
- Git (optional, for cloning)

### Installation & Setup (macOS)

```bash
# 1. Clone or navigate to the project directory
cd /path/to/your/directory

# 2. Create a virtual environment (if not already present)
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1 - M5) Note:** PyTorch MPS (GPU) acceleration is intentionally disabled in this project due to a confirmed numerical stability bug in PyTorch's MPS backend for multi-layer LSTMs. The pipeline forces CPU execution automatically — no manual configuration needed. See the Engineering Notes section below for details.

### Running the Pipeline

Once set up, run the entire pipeline end-to-end with a single command:

```bash
python3 main.py
```

The script will:
1. Scan the unzipped text directories in `Dataset/` to resolve the 62 unique daily logs.
2. Stream-parse the full 27.3 GB dataset with a constant memory increment of **0.00 MB** (baseline ~335 MB Python + PyTorch library overhead), identifying the highest-traffic cell (**Cell 5161**).
3. Extract and downcast time series for Cells `[5161, 4159, 4556]` (int16/float32 casting saves >50% RAM vs. default float64).
4. Run EDA: distribution plots, spatial heatmaps, ACF/PACF graphs, rolling stationarity tests, additive seasonal decompositions, and anomaly detection — all saved to `outputs/eda/`.
5. Force CPU execution, train PyTorch LSTM and GRU models alongside Holt-Winters, generate one-step-ahead rolling predictions on the **Dec 16 – Jan 1 test set (2,442 samples)**, measure latency, and save results to `outputs/models/`.
6. Run hyperparameter tuning experiments (3 trials × 2 models on Cell 5161), saving `outputs/models/tuning_experiments.txt`.

### Expected Runtime

| Stage | Approximate Time |
|---|---|
| Data streaming (27.3 GB) | ~5–10 min |
| EDA plots | ~2–3 min |
| Model training (3 cells × 3 models) | ~7–10 min |
| Hyperparameter tuning | ~15–20 min |
| **Total** | **~30–45 min** |

---

##   Project Directory Structure

```
ML_Pipeline/
├── Dataset/                    # !! NOT TRACKED — download manually (see Dataset section)
│   ├── dataverse_files (1)/    # Nov 1 – Nov 20
│   ├── dataverse_files (2)/    # Nov 21 – Nov 30
│   ├── dataverse_files (3)/    # Dec 1 – Dec 10
│   ├── dataverse_files (4)/    # Dec 11 – Dec 20
│   ├── dataverse_files (5)/    # Dec 21 – Dec 30
│   ├── dataverse_files (6)/    # Dec 31 – Jan 1
│   └── milano-grid.geojson     # City grid coordinates
├── outputs/
│   ├── eda/                    # Saved EDA visual plots & analysis texts
│   └── models/                 # Saved forecast overlay plots, metric summaries,
│                               #   and tuning_experiments.txt
├── src/
│   ├── data_loader.py          # Memory-efficient two-pass streaming data loaders
│   ├── eda.py                  # Core plotting and statistical test algorithms
│   ├── models.py               # PyTorch LSTM, GRU and Holt-Winters model classes
│   ├── train.py                # Model training & evaluation loops (MAE, MAPE, RMSE)
│   └── tuning.py               # Hyperparameter tuning experiments (3 trials × 2 models)
├── main.py                     # Root orchestrator running the entire pipeline
└── requirements.txt            # Project dependencies list
```

---

## Key Results

Models were evaluated on the **Dec 16 – Jan 1 test set (2,442 samples)** using MAE, MAPE, and RMSE. Each geographic cell produced a different winning model, highlighting that no single architecture dominates across all traffic regimes:

| Cell | Description | Winner | MAE | RMSE |
|---|---|---|---|---|
| **5161** | Highest traffic — city centre | Holt-Winters | 85.99 | 124.44 |
| **4159** | Moderate, variable traffic | GRU | 14.23 | 17.94 |
| **4556** | Low, uniform residential | LSTM | 26.53 | 36.36 |

---

## Engineering Notes

### 1. Memory-Efficient Two-Pass Streaming

The 27.3 GB dataset is processed line-by-line without loading it into memory. A fixed 10,001-element NumPy accumulator is used across all 62 files, achieving a **0.00 MB memory increment** during the full scan (baseline ~335 MB from Python + PyTorch library overhead).

### 2. The Apple Silicon MPS LSTM Backend Bug

During development, training on the Apple Silicon GPU backend (`mps` device in PyTorch) revealed a critical numerical stability bug in PyTorch's MPS implementation of multi-layer LSTMs:

- **On MPS:** LSTM inference diverged wildly — MAE: **732.78** on Cell 5161
- **On CPU:** Forcing CPU restored correct convergence — MAE: **123.18**

This project isolates and resolves the bug by overriding `torch.backends.mps.is_available = lambda: False` at startup. CPU execution is enforced automatically — no manual configuration needed.

### 3. Statistical vs. Deep Learning for One-Step Forecasting

Results show that no single model family dominates across all traffic regimes. Holt-Winters wins on the highest-traffic city-centre cell where strong diurnal seasonality makes local adaptive smoothing highly effective. GRU wins on moderate variable-traffic cells where sequence memory provides an edge. LSTM wins on low-traffic residential cells. This demonstrates the importance of cell-level model selection over a one-size-fits-all approach.


