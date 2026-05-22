import os
import glob
import resource
import pandas as pd
import numpy as np

def get_memory_usage_mb():
    """Returns the current resident set size (RSS) memory usage of the process in MB."""
    usage_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_bytes / (1024 * 1024)

def get_file_mapping(dataset_dir):
    """
    Scans the directories in dataset_dir and builds a unique mapping of text filename to its absolute path.
    Avoids duplicate processing of dates.
    """
    # Find all subdirectories
    subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    file_map = {}
    
    # Sort to ensure deterministic scan order
    for d in sorted(subdirs):
        # Skip the unnumbered directory to prevent duplicate processing
        if os.path.basename(d) == "dataverse_files":
            continue
        try:
            for name in os.listdir(d):
                if name.endswith(".txt"):
                    if name not in file_map:
                        file_map[name] = os.path.join(d, name)
        except Exception as e:
            print(f"Error reading directory {d}: {e}")
            
    return file_map

def compute_cell_totals(dataset_dir, file_map):
    """
    Stream-processes all files in file_map to compute the total traffic for each cell.
    Maintains a simple running sum for each cell ID to remain memory-efficient.
    """
    print("--- Phase 1: Stream Processing for Cell Traffic Sums ---")
    start_mem = get_memory_usage_mb()
    print(f"Initial Memory Usage: {start_mem:.2f} MB")
    
    # Use a flat array of size 10001 to store running sum for each square ID
    cell_totals = np.zeros(10001, dtype=np.float64)
    
    total_files = len(file_map)
    processed_count = 0
    
    for txt_name, file_path in sorted(file_map.items()):
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_files:
            print(f"  Processing file {processed_count}/{total_files}: {txt_name} ...")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\r\n').split('\t')
                if len(parts) < 8:
                    continue
                
                square_id = int(parts[0])
                internet = parts[7]
                
                if internet:  # If internet traffic is not empty
                    cell_totals[square_id] += float(internet)
                        
    end_mem = get_memory_usage_mb()
    print(f"Phase 1 Completed. Final Memory Usage: {end_mem:.2f} MB")
    print(f"Memory Increment: {end_mem - start_mem:.2f} MB")
    
    return cell_totals

def extract_target_series(dataset_dir, file_map, target_squares):
    """
    Extracts time series data for the selected target_squares from the dataset files.
    Aggregates traffic by (square_id, timestamp) by summing values across country codes.
    Uses memory downcasting to save memory.
    """
    print("\n--- Phase 2: Extracting Target Time Series ---")
    start_mem = get_memory_usage_mb()
    
    records = []
    total_files = len(file_map)
    processed_count = 0
    
    target_set = set(target_squares)
    
    for txt_name, file_path in sorted(file_map.items()):
        processed_count += 1
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\r\n').split('\t')
                if len(parts) < 8:
                    continue
                
                square_id = int(parts[0])
                if square_id in target_set:
                    timestamp = int(parts[1])
                    internet = parts[7]
                    if internet:
                        records.append({
                            'square_id': square_id,
                            'timestamp': timestamp,
                            'internet': float(internet)
                        })
                            
    # Load into pandas DataFrame
    df = pd.DataFrame(records)
    print(f"Raw target records extracted: {len(df)}")
    
    # Downcasting data types for memory efficiency
    df['square_id'] = df['square_id'].astype(np.int16)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df['internet'] = df['internet'].astype(np.float32)
    
    # Group by square_id and timestamp, summing up internet traffic across country codes
    df_agg = df.groupby(['square_id', 'timestamp'], as_index=False)['internet'].sum()
    print(f"Aggregated time series records: {len(df_agg)}")
    
    # Convert timestamp to datetime (it's in ms Unix time)
    df_agg['datetime'] = pd.to_datetime(df_agg['timestamp'], unit='ms')
    df_agg = df_agg.sort_values(by=['square_id', 'datetime']).reset_index(drop=True)
    
    end_mem = get_memory_usage_mb()
    print(f"Phase 2 Completed. Final Memory Usage: {end_mem:.2f} MB")
    print(f"Memory Increment for extraction: {end_mem - start_mem:.2f} MB")
    
    return df_agg
