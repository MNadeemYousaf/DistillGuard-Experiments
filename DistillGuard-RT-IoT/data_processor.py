import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import os
import pickle

# Configuration
DATA_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT/data"
OUTPUT_FILE = os.path.join(DATA_DIR, 'rt_iot_datasets.pt')
RAW_FILE = os.path.join(DATA_DIR, 'RT_IOT2022.csv')

# Network Features (Static/topology related)
# Note: RT-IoT here lacks IP addresses in the provided file, so we use Ports/Proto/Service
NETWORK_COLS = ['id.orig_p', 'id.resp_p', 'proto', 'service']

def process_data():
    print("Loading RT-IoT...")
    df = pd.read_csv(RAW_FILE)
    print(f"Original Shape: {df.shape}")
    
    # Drop index column (first col usually Unnamed: 0 if saved with index)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    elif df.columns[0] == '': # Empty string header
         df = df.iloc[:, 1:]
         
    # Target
    target_col = 'Attack_type'
    
    # 1. Cleaning
    print("Cleaning data...")
    # Replace Inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop NaN
    df = df.dropna()
    
    # Identify Categorical vs Numerical
    # Network cols are mixture. verify they exist
    existing_net_cols = [c for c in NETWORK_COLS if c in df.columns]
    
    # Encode Target
    y_le = LabelEncoder()
    df[target_col] = df[target_col].astype(str)
    y = y_le.fit_transform(df[target_col])
    
    # Separate Features
    X_net_raw = df[existing_net_cols].copy()
    
    # Encode Network Features
    net_encoders = {}
    for col in existing_net_cols:
        X_net_raw[col] = X_net_raw[col].astype(str)
        le = LabelEncoder()
        X_net_raw[col] = le.fit_transform(X_net_raw[col])
        net_encoders[col] = le
        
    X_net = X_net_raw.values.astype(np.float32)
    
    # Temporal Features (Candidate pools = All columns - Target - Network)
    temp_candidates = [c for c in df.columns if c not in existing_net_cols and c != target_col]
    X_temp_raw = df[temp_candidates].copy()
    
    # Handle string columns in temporal (if any)
    # Check dtypes
    obj_cols = X_temp_raw.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        print(f"Encoding additional categorical columns: {obj_cols.tolist()}")
        for col in obj_cols:
             le = LabelEncoder()
             X_temp_raw[col] = le.fit_transform(X_temp_raw[col].astype(str))
             
    # 2. Feature Selection (Mutual Information) for Temporal
    # We want Top 15 temporal features to match ToN-IoT complexity logic
    print("Selecting top 15 Temporal features using Mutual Information...")
    X_temp_np = X_temp_raw.values.astype(np.float32)
    
    # Subsample for MI if dataset is huge (RT-IoT is ~123k usually, fast enough)
    # But let's be safe
    if len(df) > 50000:
        idx = np.random.choice(len(df), 50000, replace=False)
        mi_scores = mutual_info_classif(X_temp_np[idx], y[idx], discrete_features='auto')
    else:
        mi_scores = mutual_info_classif(X_temp_np, y)
        
    top_k_indices = np.argsort(mi_scores)[-15:]
    selected_temp_cols = [temp_candidates[i] for i in top_k_indices]
    print(f"Selected Temporal Features: {selected_temp_cols}")
    
    X_temp = X_temp_np[:, top_k_indices]
    
    # 3. Normalization (MinMax)
    # Normalize Temporal (Continuous)
    scaler_temp = MinMaxScaler()
    X_temp = scaler_temp.fit_transform(X_temp)
    
    # Normalize Network? Use Embeddings?
    # DistillGuard usually embeds Network features. But our model implementation uses Linear Projections (`net_proj`).
    # Linear projection works better if inputs are normalized or if they are embeddings.
    # If we pass raw LabelEnc ints, linear layer treats them as magnitude (bad).
    # Ideally should use nn.Embedding. 
    # BUT, based on the previous `models.py` (which I copied):
    # `self.net_proj = nn.Linear(net_input_dim, hidden_dim)`
    # This implies the input `x_net` is expected to be a vector of numbers.
    # If we pass integer IDs to Linear, it learns a scalar weight for the ID.
    # For Ports/Proto, this is suboptimal but works if One-Hot encoded or Embedded.
    # Given the constraint of using the *existing* `models.py` which uses `Linear`, I should ideally One-Hot encode Network features OR MinMax scale them (heuristic).
    # Since `net_input_dim` is likely small (4), let's MinMax scale the Label IDs to [0,1].
    # This prevents exploded gradients.
    scaler_net = MinMaxScaler()
    X_net = scaler_net.fit_transform(X_net)
    
    # 4. Split 70/15/15
    X_net_train, X_net_temp, X_temp_train, X_temp_temp, y_train, y_temp = train_test_split(
        X_net, X_temp, y, test_size=0.3, stratify=y, random_state=42
    )
    X_net_val, X_net_test, X_temp_val, X_temp_test, y_val, y_test = train_test_split(
        X_net_temp, X_temp_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    ) # 0.5 of 0.3 = 0.15
    
    # 5. Save
    train_dataset = TensorDataset(torch.tensor(X_net_train), torch.tensor(X_temp_train), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_net_val), torch.tensor(X_temp_val), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_net_test), torch.tensor(X_temp_test), torch.tensor(y_test, dtype=torch.long))
    
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    torch.save(datasets, OUTPUT_FILE)
    
    # Save Encoders/Meta
    meta = {
        'net_encoders': net_encoders,
        'y_le': y_le,
        'net_cols': existing_net_cols,
        'temp_cols': selected_temp_cols
    }
    with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    print("Data processing complete. Saved datasets.")

if __name__ == "__main__":
    process_data()
