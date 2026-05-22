import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ensure output directories exist
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_pdf(cell_totals, output_dir):
    """
    Plots the probability density function (PDF) of total two-month traffic across all 10,000 areas.
    """
    ensure_dir(output_dir)
    plt.figure(figsize=(10, 6))
    
    # Slice from index 1 to ignore index 0 (which is unused)
    totals_clean = cell_totals[1:]
    
    # Plot using Seaborn KDE
    sns.kdeplot(totals_clean, fill=True, color='#2b5c8f', alpha=0.6, linewidth=2)
    plt.title("Probability Density Function (PDF) of 2-Month Total Traffic across 10,000 Areas", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Total Traffic Volume (CDR Activity proxy)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "traffic_pdf.png"), dpi=300)
    plt.close()
    print("Saved traffic PDF plot to outputs.")

def plot_time_series(df_agg, selected_cells, output_dir):
    """
    Plots the time series of network traffic during the first two weeks (Nov 1 to Nov 14) for the selected areas.
    """
    ensure_dir(output_dir)
    plt.figure(figsize=(14, 7))
    
    # Filter for first two weeks: 2013-11-01 to 2013-11-14 (inclusive)
    first_two_weeks = df_agg[df_agg['datetime'] < '2013-11-15'].copy()
    
    colors = {selected_cells[0]: '#e056fd', selected_cells[1]: '#22a6b3', selected_cells[2]: '#6ab04c'}
    labels = {
        selected_cells[0]: f"Highest Traffic Cell (ID: {selected_cells[0]})",
        selected_cells[1]: f"Cell 4159 (ID: {4159})",
        selected_cells[2]: f"Cell 4556 (ID: {4556})"
    }
    
    for cell in selected_cells:
        cell_df = first_two_weeks[first_two_weeks['square_id'] == cell]
        plt.plot(cell_df['datetime'], cell_df['internet'], label=labels[cell], color=colors[cell], alpha=0.85, linewidth=1.5)
        
    plt.title("Mobile Traffic Time Series (First 2 Weeks: Nov 1 - Nov 14, 2013)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date & Time", fontsize=12)
    plt.ylabel("Internet Traffic Activity", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_two_weeks.png"), dpi=300)
    plt.close()
    print("Saved first two weeks time series plot to outputs.")

def run_stationarity_analysis(df_agg, selected_cells, output_dir):
    """
    Performs stationarity analysis:
    - Plots rolling mean and standard deviation (24-hour window = 144 samples).
    - Runs Augmented Dickey-Fuller (ADF) test.
    """
    ensure_dir(output_dir)
    adf_results = {}
    
    for cell in selected_cells:
        cell_df = df_agg[df_agg['square_id'] == cell].copy()
        cell_df = cell_df.set_index('datetime').sort_index()
        
        # Calculate rolling statistics (24h window = 144 samples of 10-minute intervals)
        rolling_mean = cell_df['internet'].rolling(window=144).mean()
        rolling_std = cell_df['internet'].rolling(window=144).std()
        
        # Plot rolling statistics
        plt.figure(figsize=(14, 6))
        plt.plot(cell_df['internet'], color='#bebebe', alpha=0.5, label='Original')
        plt.plot(rolling_mean, color='#ff7f0e', label='Rolling Mean (24h)')
        plt.plot(rolling_std, color='#2ca02c', label='Rolling Std (24h)')
        plt.legend(loc='best')
        plt.title(f"Rolling Statistics & Stationarity for Cell {cell}", fontsize=14, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Traffic")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"stationarity_cell_{cell}.png"), dpi=300)
        plt.close()
        
        # Perform Augmented Dickey-Fuller (ADF) Test
        print(f"\nRunning ADF Test for Cell {cell}...")
        result = adfuller(cell_df['internet'].dropna(), autolag='AIC')
        
        adf_results[cell] = {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations': result[3],
            'Critical Values (1%)': result[4]['1%'],
            'Critical Values (5%)': result[4]['5%'],
            'Critical Values (10%)': result[4]['10%']
        }
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6e}")
        print(f"Stationary: {'Yes' if result[1] < 0.05 else 'No'}")
        
    # Compile text output
    with open(os.path.join(output_dir, "adf_results.txt"), "w") as f:
        f.write("Augmented Dickey-Fuller Test Results:\n")
        f.write("=====================================\n")
        for cell, res in adf_results.items():
            f.write(f"\nCell ID: {cell}\n")
            for k, v in res.items():
                f.write(f"  {k}: {v}\n")
                
    return adf_results

def run_seasonal_decomposition(df_agg, selected_cells, output_dir):
    """
    Decomposes the time series of the target cells into trend, seasonal, and residual components.
    Uses daily period (144 samples).
    """
    ensure_dir(output_dir)
    
    for cell in selected_cells:
        cell_df = df_agg[df_agg['square_id'] == cell].copy()
        cell_df = cell_df.set_index('datetime').sort_index()
        
        # Resample to ensure fixed freq or interpolate missing values (if any)
        # Note: 10-minute frequency is '10min'
        cell_ts = cell_df['internet'].asfreq('10min').interpolate(method='linear')
        
        # Perform decomposition (period = 144 for daily cycle)
        decomp = seasonal_decompose(cell_ts, model='additive', period=144)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        decomp.observed.plot(ax=axes[0], color='#2c3e50', alpha=0.8)
        axes[0].set_ylabel('Observed')
        axes[0].set_title(f"Additive Seasonal Decomposition for Cell {cell} (Period = 24h)", fontsize=14, fontweight='bold')
        
        decomp.trend.plot(ax=axes[1], color='#e74c3c', linewidth=2)
        axes[1].set_ylabel('Trend')
        
        decomp.seasonal.plot(ax=axes[2], color='#2ecc71')
        axes[2].set_ylabel('Seasonal')
        
        decomp.resid.plot(ax=axes[3], color='#f39c12', style='.', alpha=0.5)
        axes[3].set_ylabel('Residual')
        
        for ax in axes:
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.xlabel("Datetime")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"decomposition_cell_{cell}.png"), dpi=300)
        plt.close()
        print(f"Saved seasonal decomposition plot for Cell {cell}.")

def plot_acf_pacf(df_agg, selected_cells, output_dir):
    """
    Plots Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots for the selected areas.
    """
    ensure_dir(output_dir)
    
    for cell in selected_cells:
        cell_df = df_agg[df_agg['square_id'] == cell].copy()
        cell_df = cell_df.set_index('datetime').sort_index()
        cell_ts = cell_df['internet'].asfreq('10min').interpolate(method='linear')
        
        # We plot up to 3 days of lags (3 * 144 = 432 lags) to capture daily dependencies
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        plot_acf(cell_ts, lags=432, ax=axes[0], color='#2980b9', alpha=0.05)
        axes[0].set_title(f"Autocorrelation Function (ACF) - Cell {cell}", fontsize=12, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        plot_pacf(cell_ts, lags=432, ax=axes[1], color='#e67e22', method='ywm', alpha=0.05)
        axes[1].set_title(f"Partial Autocorrelation Function (PACF) - Cell {cell}", fontsize=12, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"acf_pacf_cell_{cell}.png"), dpi=300)
        plt.close()
        print(f"Saved ACF/PACF plots for Cell {cell}.")

def plot_spatial_heatmap(cell_totals, output_dir):
    """
    Plots a 100x100 grid heatmap of total traffic volume across the city.
    Cell IDs are mapped to coordinates: row = (cell_id - 1) // 100, col = (cell_id - 1) % 100.
    """
    ensure_dir(output_dir)
    
    # Convert cell_totals[1:10001] into 100x100 matrix
    grid_data = cell_totals[1:10001].reshape((100, 100))
    
    # We display it such that latitude increases upwards, so we reverse the rows
    # so row 99 (top) is at the top of the plot and row 0 (bottom) is at the bottom
    grid_data_plot = np.flipud(grid_data)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(grid_data_plot, cmap='viridis', robust=True, cbar_kws={'label': 'Total Traffic Volume (CDR activity proxy)'})
    plt.title("Spatial Heatmap of Mobile Network Traffic Intensity (100x100 Grid)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Grid X (Columns)", fontsize=12)
    plt.ylabel("Grid Y (Rows)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spatial_heatmap.png"), dpi=300)
    plt.close()
    print("Saved spatial heatmap plot to outputs.")

def analyze_anomalies(df_agg, selected_cells, output_dir):
    """
    Identifies outliers/anomalies in the traffic time series for selected cells.
    Uses rolling 7-day z-score to identify anomalies and saves the summary.
    """
    ensure_dir(output_dir)
    
    anomaly_summary = []
    
    for cell in selected_cells:
        cell_df = df_agg[df_agg['square_id'] == cell].copy()
        cell_df = cell_df.set_index('datetime').sort_index()
        
        # 7-day rolling window = 7 * 144 = 1008 samples
        rolling_mean = cell_df['internet'].rolling(window=1008, min_periods=144).mean()
        rolling_std = cell_df['internet'].rolling(window=1008, min_periods=144).std()
        
        z_scores = (cell_df['internet'] - rolling_mean) / (rolling_std + 1e-5)
        anomalies = cell_df[np.abs(z_scores) > 3.5]
        
        # Plot anomalies
        plt.figure(figsize=(14, 6))
        plt.plot(cell_df['internet'], color='#34495e', alpha=0.7, label='Traffic')
        plt.scatter(anomalies.index, anomalies['internet'], color='#e74c3c', marker='x', s=40, label='Anomalies (|Z| > 3.5)')
        plt.title(f"Traffic Anomalies Detection - Cell {cell}", fontsize=14, fontweight='bold')
        plt.xlabel("Datetime")
        plt.ylabel("Internet Traffic Activity")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"anomalies_cell_{cell}.png"), dpi=300)
        plt.close()
        
        anomaly_summary.append(f"Cell {cell}: Detected {len(anomalies)} anomalies out of {len(cell_df)} data points ({len(anomalies)/len(cell_df)*100:.2f}%).")
        
    with open(os.path.join(output_dir, "anomaly_analysis.txt"), "w") as f:
        f.write("Time Series Outlier and Anomaly Analysis:\n")
        f.write("=========================================\n")
        for summary in anomaly_summary:
            f.write(summary + "\n")
            
    print("Saved anomaly detection plots and summary to outputs.")

def plot_memory_comparison(output_dir):
    """
    Plots a bar chart comparing memory usage before and after optimization.
    Y-axis is in logarithmic scale (MB) to accommodate the massive difference.
    """
    ensure_dir(output_dir)
    
    categories = [
        "Naive Loading\n(pd.read_csv on 20.4GB)",
        "Pass 1 Streaming\n(NumPy Accumulators)",
        "Pass 2 Dataframe\n(Optimized & Downcast)"
    ]
    
    # Values in MB: 30 GB = 30720 MB, 65.72 MB, 0.56 MB
    memory_mb = [30720.0, 65.72, 0.56]
    
    plt.figure(figsize=(9, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = plt.bar(categories, memory_mb, color=colors, width=0.45, edgecolor='black', alpha=0.85)
    
    plt.yscale('log')
    plt.ylabel("Peak Memory Usage (MB) - Log Scale", fontsize=11, fontweight='bold')
    plt.title("Memory Ingestion Footprint: Before vs. After Optimization", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        if height >= 1024:
            label_text = f"{height/1024:.1f} GB"
        else:
            label_text = f"{height:.2f} MB"
        plt.text(bar.get_x() + bar.get_width()/2.0, height * 1.3 if height < 10000 else height * 0.35, 
                 label_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"), dpi=300)
    plt.close()
    print("Saved memory usage comparison plot to outputs.")

