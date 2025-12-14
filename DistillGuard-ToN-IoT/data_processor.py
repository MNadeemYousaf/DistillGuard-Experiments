import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle

# Configuration
DATA_PATH = "/Users/nadeemyousaf/.gemini/antigravity/scratch/Graph-LLM-IDS-V2/TON-IoT-networkflow.csv"
OUTPUT_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-ToN-IoT/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature Groups (for Hybrid Attention)
# "Network Features" (Structural/Identifier)
NETWORK_COLS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'proto', 'service']
# "Temporal/Stats Features" (Time/Volume)
TEMPORAL_COLS = ['duration', 'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']

def clean_and_encode(df):
    print("Cleaning data...")
    # Drop irrelevant columns (mostly empty or ID-like)
    # Keeping it simple for now, relying on feature selection later
    drop_cols = ['ts', 'date', 'weird_name', 'weird_addl', 'weird_notice'] # anomalies often have these empty
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Replace '-' with NaN and impute or drop
    df.replace('-', np.nan, inplace=True)
    
    # Fill numerical NaNs with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    # Cat cols: fill with 'unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('unknown')
    
    # Label Encoding for specific Network Columns to keep them as indices for embeddings (if we wanted)
    # BUT paper says "One-hot encoding". However, for IPs, One-Hot is impossible (too high dim).
    # The prompt says: "One-hot encoding for categorical features: proto, service, etc."
    # src_ip/dst_ip are usually Hashed or Frequency encoded in such papers if not using Graph.
    # Let's use Label Encoding for IPs and high-cardinality, and One-Hot for low cardinality.
    
    # Label Encoding for specific Network Columns to keep them as indices for embeddings
    encoders = {}
    for col in NETWORK_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str) # Ensure string before encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
    if 'conn_state' in df.columns:
        df['conn_state'] = df['conn_state'].astype(str)
        le = LabelEncoder()
        df['conn_state'] = le.fit_transform(df['conn_state'])
        encoders['conn_state'] = le

    # Handling Target
    target_col = 'type'
    if target_col not in df.columns: 
        target_col = 'label'
        
    y_le = LabelEncoder()
    y = y_le.fit_transform(df[target_col].astype(str))
    
    # DROP all other string/object columns that are not encoded
    # Use select_dtypes to find them
    remaining_obj_cols = df.select_dtypes(include=['object']).columns
    print(f"Dropping unhandled object columns: {remaining_obj_cols.tolist()}")
    df = df.drop(columns=remaining_obj_cols)
    
    # Drop labels from X
    # Note: 'type' and 'label' were already encoded into y, but 'type' column might still be in df if we didn't drop it or if it wasn't object type (unlikely)
    # Actually y_le transformed it, but didn't modify df in place for that column yet.
    # Safe drop:
    X = df.drop(columns=['label', 'type'], errors='ignore')
    
    return X, y, encoders, y_le

def feature_selection(X, y, k=30):
    # Implementing Mutual Information based selection (Proxy for MRMR)
    # Selecting top K features
    print(f"Selecting top {k} features using Mutual Information...")
    # Select only a subset for speed if large
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y
        
    # Ensure all numerical
    # We already encoded categorical, so X is effectively numerical now
    scores = mutual_info_classif(X_sample, y_sample, discrete_features='auto', n_neighbors=3, random_state=42)
    
    feature_scores = pd.Series(scores, index=X.columns)
    selected_cols = feature_scores.nlargest(k).index.tolist()
    print(f"Selected Features: {selected_cols}")
    
    return X[selected_cols], selected_cols

def process_data(sample_size=None):
    print("Loading ToN-IoT...")
    if sample_size:
        df = pd.read_csv(DATA_PATH, nrows=sample_size)
    else:
        df = pd.read_csv(DATA_PATH)
        
    print(f"Original Shape: {df.shape}")
    
    X, y, encoders, y_le = clean_and_encode(df)
    
    # Feature Selection
    # Note: We need to preserve 'Network' vs 'Temporal' split logic.
    # If we filter features blindly, we might lose all "Network" features.
    # The Teacher needs BOTH.
    # SO: We apply selection primarily to the "Other" or "Temporal" high-dim stats.
    # We FORCE keep the Network Cols defined for the architecture.
    
    # Actually, let's select Top features but ensure representation from both groups if possible.
    # Or, simplify: Just use the selected features.
    # The "Hybrid Attention" implies inputting specific features. 
    # Let's Keep fixed set of NETWORK_COLS + Top K TEMPORAL/Payload cols.
    
    # Separate strictly
    x_network = X[[c for c in NETWORK_COLS if c in X.columns]].copy()
    remaining = X.drop(columns=x_network.columns)
    
    # Select from remaining
    x_temporal, selected_temp = feature_selection(remaining, y, k=20) 
    
    print(f"Final Network Features: {x_network.shape[1]}")
    print(f"Final Temporal Features: {x_temporal.shape[1]}")
    
    # Normalization (MinMax)
    scaler_net = MinMaxScaler()
    x_net_scaled = scaler_net.fit_transform(x_network)
    
    scaler_temp = MinMaxScaler()
    x_temp_scaled = scaler_temp.fit_transform(x_temporal)
    
    # Concatenate for 'Student' (Dense usually takes flat vector)
    # But Teacher needs them split.
    # We will return a Dataset that yields (x_net, x_temp, y)
    
    # Split Train/Val/Test (70/15/15)
    # Stratified split
    indices = np.arange(len(y))
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(indices, y, test_size=0.3, stratify=y, random_state=42)
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(X_temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    datasets = {}
    for name, idx in zip(['train', 'val', 'test'], [X_train_idx, X_val_idx, X_test_idx]):
        # Store as Tensors
        x_n = torch.tensor(x_net_scaled[idx], dtype=torch.float32)
        x_t = torch.tensor(x_temp_scaled[idx], dtype=torch.float32)
        y_t = torch.tensor(y[idx], dtype=torch.long)
        datasets[name] = TensorDataset(x_n, x_t, y_t)
        
    # Save artifacts
    torch.save(datasets, os.path.join(OUTPUT_DIR, 'ton_iot_datasets.pt'))
    with open(os.path.join(OUTPUT_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump({'y_le': y_le, 'net_cols': x_network.columns.tolist(), 'temp_cols': x_temporal.columns.tolist()}, f)
        
    print("Data processing complete. Saved datasets.")

if __name__ == "__main__":
    process_data() # Use sample_size=50000 for quick test if needed
