import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import pickle
import glob

# Configuration
DATA_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-N-BaIoT/data"
OUTPUT_FILE = os.path.join(DATA_DIR, 'n_baiot_datasets.pt')
RAW_DIR = DATA_DIR

def process_data():
    print("Loading N-BaIoT Full Dataset (All Devices) - Optimized...")
    
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    files = [f for f in files if 'features.csv' not in f and 'data_summary.csv' not in f and 'device_info.csv' not in f]
    
    dfs = []
    print(f"Found {len(files)} files. Sampling ~3% from each to fit memory...")
    
    # Target total ~200k. 89 files. ~2200 rows per file.
    # Or percentage based.
    
    for f in files:
        fname = os.path.basename(f)
        # Parse label
        parts = fname.split('.')
        if len(parts) >= 3:
            label = "_".join(parts[1:-1])
        else:
            label = "benign"
            
        try:
            # Read only a fraction (e.g. 5%)
            # Using skiprows=lambda x: x > 0 and np.random.rand() > 0.05 is slow.
            # Better: Read full file? No, memory issues.
            # Best: Read n rows? No, bias to start.
            # Read with sample? Pandas read_csv doesn't sample natively without reading.
            # Workaround: Read chunks and sample.
            chunk_size = 50000
            file_dfs = []
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                # Sample 3%
                sample = chunk.sample(frac=0.03)
                file_dfs.append(sample)
                
            df_chunk = pd.concat(file_dfs)
            df_chunk['Label'] = label
            dfs.append(df_chunk)
            print(f"Processed {fname} -> {len(df_chunk)} rows")
            
        except Exception as e:
            print(f"Skipping {fname}: {e}")
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total Subsampled Shape: {df.shape}")
    
    # Target Processing
    target_col = 'Label'
    y_le = LabelEncoder()
    y = y_le.fit_transform(df[target_col])
    
    all_features = [c for c in df.columns if c != target_col]
    
    # Hybrid Split
    net_cols = [c for c in all_features if 'L1' in c or 'L3' in c]
    temp_cols = [c for c in all_features if c not in net_cols]
    
    print(f"Split Features -> Net: {len(net_cols)}, Temp: {len(temp_cols)}")
    
    X_net = df[net_cols].values.astype(np.float32)
    X_temp = df[temp_cols].values.astype(np.float32)
    
    # Normalization
    scaler_net = MinMaxScaler()
    X_net = scaler_net.fit_transform(X_net)
    
    scaler_temp = MinMaxScaler()
    X_temp = scaler_temp.fit_transform(X_temp)
    
    # Split 70/15/15
    X_net_train, X_net_temp, X_temp_train, X_temp_temp, y_train, y_temp = train_test_split(
        X_net, X_temp, y, test_size=0.3, stratify=y, random_state=42
    )
    X_net_val, X_net_test, X_temp_val, X_temp_test, y_val, y_test = train_test_split(
        X_net_temp, X_temp_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    meta = {
        'y_le': y_le,
        'net_cols': net_cols,
        'temp_cols': temp_cols
    }
    with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    datasets = {
        'train': TensorDataset(torch.tensor(X_net_train), torch.tensor(X_temp_train), torch.tensor(y_train)),
        'val': TensorDataset(torch.tensor(X_net_val), torch.tensor(X_temp_val), torch.tensor(y_val)),
        'test': TensorDataset(torch.tensor(X_net_test), torch.tensor(X_temp_test), torch.tensor(y_test))
    }
    torch.save(datasets, OUTPUT_FILE)
    print("Data processing complete.")

if __name__ == "__main__":
    process_data()
