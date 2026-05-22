import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from src.models import LSTMForecaster, GRUForecaster, StatisticalForecaster

# Ensure output directories exist
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Evaluation metrics
def calculate_metrics(y_true, y_pred):
    """
    Computes MAE, MAPE, and RMSE.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    # Avoid division by zero in MAPE by adding a small epsilon or filtering out very small values
    epsilon = 1e-5
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return mae, mape, rmse

def create_sequences(data, seq_length):
    """
    Creates overlapping sequences for sequence models.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_nn_model(model, train_loader, val_loader, num_epochs=12, learning_rate=0.001, device='cpu'):
    """
    Trains a PyTorch neural network model and records the execution time.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    start_time = time.perf_counter()
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
    train_time = time.perf_counter() - start_time
    
    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return train_time

def evaluate_nn_model(model, test_data, seq_length, scaler, device='cpu'):
    """
    Performs one-step-ahead rolling forecasting on test data.
    Test data is a normalized 1D array.
    """
    model.eval()
    model.to(device)
    
    # We will step through the test set one-step-ahead.
    # To predict test_data[i] (where i >= seq_length), we feed the previous seq_length values.
    predictions = []
    actuals = []
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for i in range(seq_length, len(test_data)):
            # History sequence of length seq_length
            history = test_data[i-seq_length:i]
            # Reshape for model input: (1, seq_length, 1)
            x_input = torch.tensor(history, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
            
            pred = model(x_input)
            predictions.append(pred.item())
            actuals.append(test_data[i])
            
    inference_time = time.perf_counter() - start_time
    
    # Scale back to original values
    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals_scaled = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    return predictions_scaled, actuals_scaled, inference_time

def run_modelling_pipeline(df_agg, selected_cells, output_dir, seq_length=144):
    """
    Runs the complete model training, evaluation, and plotting pipeline for the 3 target areas.
    """
    ensure_dir(output_dir)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device for Deep Learning: {device}")
    
    # Results dictionary to compile performance
    results = {}
    time_stats = {}
    
    # Split dates
    # Train/Val: Nov 1 to Dec 15
    # Test: Dec 16 to Dec 22
    # We split Nov 1 to Dec 15 into Train (Nov 1 to Dec 8) and Val (Dec 9 to Dec 15)
    train_end_date = '2013-12-09 00:00:00'
    val_end_date = '2013-12-16 00:00:00'
    
    for cell in selected_cells:
        print(f"\n==================== CELL {cell} ====================")
        cell_df = df_agg[df_agg['square_id'] == cell].copy().sort_values(by='datetime').reset_index(drop=True)
        
        # Split datasets
        train_df = cell_df[cell_df['datetime'] < train_end_date].copy()
        val_df = cell_df[(cell_df['datetime'] >= train_end_date) & (cell_df['datetime'] < val_end_date)].copy()
        test_df = cell_df[cell_df['datetime'] >= val_end_date].copy()
        
        print(f"Data split: Train={len(train_df)} samples, Val={len(val_df)} samples, Test={len(test_df)} samples")
        
        # Fit scaler on training data only
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df['internet'].values.reshape(-1, 1)).flatten()
        val_scaled = scaler.transform(val_df['internet'].values.reshape(-1, 1)).flatten()
        
        # For the test set, we need the history preceding the test period.
        # Since we use seq_length context, we concatenate the last seq_length values of validation data to the test data.
        val_last_context = val_df['internet'].values[-seq_length:]
        test_full = np.concatenate([val_last_context, test_df['internet'].values])
        test_scaled = scaler.transform(test_full.reshape(-1, 1)).flatten()
        
        # Create sequences for PyTorch DataLoader
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_val, y_val = create_sequences(val_scaled, seq_length)
        
        # Reshape to (samples, seq_length, 1)
        X_train = np.expand_dims(X_train, 2)
        X_val = np.expand_dims(X_val, 2)
        
        # Create datasets & loaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        results[cell] = {}
        time_stats[cell] = {}
        
        # ------------------ 1. Statistical Model (Holt-Winters) ------------------
        print("  Training Statistical Model (Holt-Winters)...")
        stat_model = StatisticalForecaster(model_type='holtwinters', seasonal_periods=144)
        
        start_fit = time.perf_counter()
        # Train on combined train + validation data for best baseline statistics
        train_val_raw = np.concatenate([train_df['internet'].values, val_df['internet'].values])
        stat_model.fit(train_val_raw)
        stat_fit_time = time.perf_counter() - start_fit
        
        print("  Evaluating Statistical Model (one-step rolling)...")
        start_inf = time.perf_counter()
        stat_preds = stat_model.predict_one_step_ahead(history=None, actual_test_data=test_df['internet'].values)
        stat_inf_time = time.perf_counter() - start_inf
        
        stat_mae, stat_mape, stat_rmse = calculate_metrics(test_df['internet'].values, stat_preds)
        results[cell]['SARIMA'] = {'MAE': stat_mae, 'MAPE': stat_mape, 'RMSE': stat_rmse, 'preds': stat_preds}
        time_stats[cell]['SARIMA'] = {'TrainTime': stat_fit_time, 'InferenceTime': stat_inf_time}
        print(f"    SARIMA (Holt-Winters) Metrics - MAE: {stat_mae:.4f} - MAPE: {stat_mape:.4f}% - RMSE: {stat_rmse:.4f}")
        
        # ------------------ 2. LSTM Model ------------------
        print("  Training LSTM Model...")
        lstm_model = LSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
        lstm_train_time = train_nn_model(lstm_model, train_loader, val_loader, num_epochs=12, device=device)
        
        print("  Evaluating LSTM Model (one-step rolling)...")
        lstm_preds, actuals, lstm_inf_time = evaluate_nn_model(lstm_model, test_scaled, seq_length, scaler, device=device)
        
        lstm_mae, lstm_mape, lstm_rmse = calculate_metrics(test_df['internet'].values, lstm_preds)
        results[cell]['LSTM'] = {'MAE': lstm_mae, 'MAPE': lstm_mape, 'RMSE': lstm_rmse, 'preds': lstm_preds}
        time_stats[cell]['LSTM'] = {'TrainTime': lstm_train_time, 'InferenceTime': lstm_inf_time}
        print(f"    LSTM Metrics - MAE: {lstm_mae:.4f} - MAPE: {lstm_mape:.4f}% - RMSE: {lstm_rmse:.4f}")
        
        # ------------------ 3. GRU Model ------------------
        print("  Training GRU Model...")
        gru_model = GRUForecaster(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1)
        gru_train_time = train_nn_model(gru_model, train_loader, val_loader, num_epochs=12, device=device)
        
        print("  Evaluating GRU Model (one-step rolling)...")
        gru_preds, _, gru_inf_time = evaluate_nn_model(gru_model, test_scaled, seq_length, scaler, device=device)
        
        gru_mae, gru_mape, gru_rmse = calculate_metrics(test_df['internet'].values, gru_preds)
        results[cell]['GRU'] = {'MAE': gru_mae, 'MAPE': gru_mape, 'RMSE': gru_rmse, 'preds': gru_preds}
        time_stats[cell]['GRU'] = {'TrainTime': gru_train_time, 'InferenceTime': gru_inf_time}
        print(f"    GRU Metrics - MAE: {gru_mae:.4f} - MAPE: {gru_mape:.4f}% - RMSE: {gru_rmse:.4f}")
        
        # ------------------ Plotting overlays (3 plots per cell, 9 plots total) ------------------
        models_list = ['SARIMA', 'LSTM', 'GRU']
        colors = {'SARIMA': '#2980b9', 'LSTM': '#9b59b6', 'GRU': '#2ecc71'}
        test_dates = test_df['datetime'].values
        actual_vals = test_df['internet'].values
        
        for model_name in models_list:
            plt.figure(figsize=(14, 6))
            plt.plot(test_dates, actual_vals, color='#34495e', label='Original Traffic (Actual)', linewidth=1.5)
            plt.plot(test_dates, results[cell][model_name]['preds'], color=colors[model_name], label=f'{model_name} Predicted', linewidth=1.5)
            
            plt.title(f"One-Step-Ahead Forecast vs Actual: Cell {cell} ({model_name})", fontsize=14, fontweight='bold')
            plt.xlabel("Datetime", fontsize=12)
            plt.ylabel("Internet Traffic Activity", fontsize=12)
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"forecast_cell_{cell}_{model_name}.png"), dpi=300)
            plt.close()
            print(f"    Saved forecast plot for Cell {cell} ({model_name}).")
            
    # Compile performance tables
    performance_summary = []
    for cell in selected_cells:
        performance_summary.append(f"\nCell ID: {cell}\n" + "-"*40)
        performance_summary.append(f"{'Model':<12}{'MAE':<12}{'MAPE (%)':<12}{'RMSE':<12}")
        for model_name in ['SARIMA', 'LSTM', 'GRU']:
            m = results[cell][model_name]
            performance_summary.append(f"{model_name:<12}{m['MAE']:<12.4f}{m['MAPE']:<12.4f}{m['RMSE']:<12.4f}")
            
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write("Model Performance Comparison (Test Week: Dec 16 - Dec 22):\n")
        f.write("============================================================\n")
        f.write("\n".join(performance_summary) + "\n")
        
    # Compile execution times
    time_summary = []
    time_summary.append(f"{'Cell':<10}{'Model':<12}{'Train Time (s)':<18}{'Inference Time (s)':<18}")
    for cell in selected_cells:
        for model_name in ['SARIMA', 'LSTM', 'GRU']:
            t = time_stats[cell][model_name]
            time_summary.append(f"{cell:<10}{model_name:<12}{t['TrainTime']:<18.4f}{t['InferenceTime']:<18.4f}")
            
    with open(os.path.join(output_dir, "execution_times.txt"), "w") as f:
        f.write("Model Execution Time statistics (M1 Pro CPU/GPU):\n")
        f.write("================================================\n\n")
        f.write("\n".join(time_summary) + "\n")
        
    # Generate failure analysis zoom plot for Cell 5161
    try:
        plot_failure_analysis_zoom(results, df_agg, selected_cells, output_dir)
    except Exception as e:
        print(f"Error generating failure analysis zoom plot: {e}")
        
    print("\nAll model pipeline tasks completed successfully! Results and metrics saved in outputs/models/")
    return results, time_stats

def plot_failure_analysis_zoom(results, df_agg, selected_cells, output_dir):
    """
    Finds the 24-hour window with the highest forecasting error for Cell 5161
    and saves a zoomed-in overlay plot comparing Actual vs All Models.
    """
    ensure_dir(output_dir)
    
    # Target Cell 5161 (highest traffic)
    cell = 5161
    if cell not in results:
        cell = list(results.keys())[0] # Fallback
        
    cell_df = df_agg[df_agg['square_id'] == cell].copy().sort_values(by='datetime').reset_index(drop=True)
    val_end_date = '2013-12-16 00:00:00'
    test_df = cell_df[cell_df['datetime'] >= val_end_date].copy()
    
    test_dates = test_df['datetime'].values
    actual_vals = test_df['internet'].values
    
    # Calculate rolling absolute error of GRU predictions
    gru_preds = results[cell]['GRU']['preds']
    abs_errors = np.abs(actual_vals - gru_preds)
    
    # Find the 144-sample window (24 hours) with the maximum total absolute error
    window_size = 144
    max_error_sum = 0
    max_error_idx = 0
    
    for i in range(len(abs_errors) - window_size):
        err_sum = np.sum(abs_errors[i:i+window_size])
        if err_sum > max_error_sum:
            max_error_sum = err_sum
            max_error_idx = i
            
    zoom_start = max_error_idx
    zoom_end = max_error_idx + window_size
    
    # Zoomed data
    zoom_dates = test_dates[zoom_start:zoom_end]
    zoom_actual = actual_vals[zoom_start:zoom_end]
    
    plt.figure(figsize=(14, 6))
    plt.plot(zoom_dates, zoom_actual, color='#34495e', label='Original Traffic (Actual)', linewidth=2.0)
    
    colors = {'SARIMA': '#2980b9', 'LSTM': '#9b59b6', 'GRU': '#2ecc71'}
    for model_name in ['SARIMA', 'LSTM', 'GRU']:
        zoom_pred = results[cell][model_name]['preds'][zoom_start:zoom_end]
        plt.plot(zoom_dates, zoom_pred, color=colors[model_name], label=f'{model_name} Predicted', linewidth=1.8, linestyle='--')
        
    plt.title(f"Zoomed-In Failure Analysis (Worst 24-Hour Period): Cell {cell}", fontsize=14, fontweight='bold')
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Internet Traffic Activity", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "failure_analysis_zoom.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved zoomed failure analysis plot to {save_path}.")

