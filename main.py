import os
import sys

# Ensure project directories are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
import torch
# Force CPU device for training/evaluation to bypass the PyTorch Apple Silicon MPS LSTM backend bug
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: False

from src.data_loader import get_file_mapping, compute_cell_totals, extract_target_series
from src.eda import (
    plot_pdf,
    plot_time_series,
    run_stationarity_analysis,
    run_seasonal_decomposition,
    plot_acf_pacf,
    plot_spatial_heatmap,
    analyze_anomalies,
    plot_memory_comparison
)
from src.train import run_modelling_pipeline
from src.tuning import run_hyperparameter_tuning

def main():
    print("==============================================================")
    # 0. Configuration
    dataset_dir = os.path.abspath("Dataset")
    eda_output_dir = os.path.abspath("outputs/eda")
    models_output_dir = os.path.abspath("outputs/models")
    
    print(f"Dataset path: {dataset_dir}")
    print(f"EDA Outputs path: {eda_output_dir}")
    print(f"Models Outputs path: {models_output_dir}")
    print("==============================================================\n")
    
    # 1. File mapping
    print("Mapping files...")
    file_map = get_file_mapping(dataset_dir)
    print(f"Mapped {len(file_map)} files from dataset subdirectories.")
    
    # 2. Cell traffic summation (Pass 1)
    cell_totals = compute_cell_totals(dataset_dir, file_map)
    highest_cell = int(cell_totals.argmax())
    print(f"Highest traffic cell: {highest_cell} with total traffic: {cell_totals[highest_cell]:.4f}")
    
    # 3. Target series extraction (Pass 2)
    selected_cells = [highest_cell, 4159, 4556]
    print(f"Extracting time series for cells: {selected_cells}...")
    df_agg = extract_target_series(dataset_dir, file_map, selected_cells)
    
    # 4. Exploratory Data Analysis (EDA)
    print("\n--- Running Exploratory Data Analysis (EDA) ---")
    plot_pdf(cell_totals, eda_output_dir)
    plot_spatial_heatmap(cell_totals, eda_output_dir)
    plot_time_series(df_agg, selected_cells, eda_output_dir)
    run_stationarity_analysis(df_agg, selected_cells, eda_output_dir)
    run_seasonal_decomposition(df_agg, selected_cells, eda_output_dir)
    plot_acf_pacf(df_agg, selected_cells, eda_output_dir)
    analyze_anomalies(df_agg, selected_cells, eda_output_dir)
    plot_memory_comparison(eda_output_dir)
    print("EDA completed successfully.")
    
    # 4.5 Hyperparameter Tuning (Grid Search & Narrative Documentation)
    print("\n--- Running Systematic Hyperparameter Tuning (Cell 5161) ---")
    run_hyperparameter_tuning(df_agg, highest_cell, models_output_dir)
    print("Hyperparameter tuning completed successfully.")
    
    # 5. Modeling Pipeline (SARIMA/Holt-Winters, LSTM, GRU)
    print("\n--- Running Model Training and Evaluation Pipeline (CPU Forced) ---")
    results, time_stats = run_modelling_pipeline(df_agg, selected_cells, models_output_dir, seq_length=144)
    print("Modeling pipeline completed successfully.")

if __name__ == "__main__":
    main()
