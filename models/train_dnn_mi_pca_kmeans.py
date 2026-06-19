import pandas as pd
import numpy as np
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

# Safe automated installation/import layer for imbalanced-learn utilities
try:
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("Installing missing imbalanced-learn packages...")
    os.system('pip install imbalanced-learn')
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler

# Set deterministic seeds for deep learning reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using execution device: {device}")

# =========================================================================
# === PHASE 3: PRODUCTION PREPROCESSING LAYER ===
# =========================================================================
print("\n=== Phase 3: Commencing Preprocessing Layer ===")

col_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
    'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_src_ltm', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm_d', 'ct_srv_dst', 'ct_state_ttl', 'ct_src_user_ltm',
    'ct_src_zone_ltm', 'ct_dst_host_ltm', 'ct_srv_src', 'ct_dst_sport_ltm_d', 'ct_dst_src_ltm_d',
    'ct_src_ltm_d_d', 'ct_src_ltm_d_s', 'ct_dst_ltm_d_d', 'ct_dst_ltm_d_s', 'ct_srv_dst_d', 'ct_srv_src_d',
    'ct_state_ttl_d', 'ct_src_user_ltm_d', 'ct_src_zone_ltm_d', 'ct_dst_host_ltm_d', 'ct_srv_dst_d_d',
    'ct_srv_src_d_d', 'ct_state_ttl_d_d', 'ct_src_user_ltm_d_d', 'ct_src_zone_ltm_d_d', 'ct_dst_host_ltm_d_d',
    'ct_srv_dst_d_d_d', 'ct_srv_src_d_d_d', 'ct_state_ttl_d_d_d', 'ct_src_user_ltm_d_d_d',
    'ct_src_zone_ltm_d_d_d', 'ct_dst_host_ltm_d_d_d', 'id', 'attack_cat', 'label'
]

files = [f'UNSW-NB15_{i}.csv' for i in range(1, 4 + 1)]
df_list = []

for f in files:
    try:
        print(f"Loading {f}...")
        df_temp = pd.read_csv(f, header=None, low_memory=False)
        if df_temp.shape[1] == 49:
            df_temp.columns = col_names[:47] + ['attack_cat', 'label']
        else:
            df_temp.columns = col_names[:df_temp.shape[1]]
        df_list.append(df_temp)
    except FileNotFoundError:
        print(f"⚠️ Warning: {f} not found.")

df = pd.concat(df_list, ignore_index=True)
print(f"Data Ingestion Complete. Combined Shape: {df.shape}")

# Target Processing & Standardization Mapping
target_col = 'attack_cat'
df[target_col] = df[target_col].fillna('Normal').astype(str).str.strip().str.lower()

category_mapping = {
    'normal': 'Normal', 'fuzzers': 'Fuzzers', 'analysis': 'Analysis',
    'backdoor': 'Backdoor', 'dos': 'DoS', 'exploits': 'Exploits',
    'generic': 'Generic', 'reconnaissance': 'Reconnaissance',
    'shellcode': 'Shellcode', 'worms': 'Worms'
}
df[target_col] = df[target_col].map(category_mapping).fillna('Normal')

target_encoder = LabelEncoder()
y_all = target_encoder.fit_transform(df[target_col])
num_classes = len(target_encoder.classes_)
normal_class_idx = list(target_encoder.classes_).index('Normal')

drop_cols = ['id', 'label', 'stime', 'ltime', 'srcip', 'dstip']
X_raw = df.drop([c for c in drop_cols if c in df.columns] + [target_col], axis=1)
del df, df_list; gc.collect()

# --- Continuous Feature Log-Transformation Layer ---
for port_col in ['sport', 'dsport']:
    if port_col in X_raw.columns:
        X_raw[port_col] = pd.to_numeric(X_raw[port_col], errors='coerce').fillna(-1).astype('int32')

true_cat_features = ['proto', 'state', 'service']
binary_features = ['is_ftp_login', 'is_sm_ips_ports']
continuous_features = [col for col in X_raw.columns if col not in true_cat_features + binary_features]

X_continuous = np.log1p(X_raw[continuous_features].clip(lower=0)).fillna(0).astype('float32')

X_categorical = pd.DataFrame(index=X_raw.index)
for col in true_cat_features:
    if col in X_raw.columns:
        X_categorical[col] = LabelEncoder().fit_transform(X_raw[col].astype(str)).astype('float32')

X_binary = X_raw[binary_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

X_processed = np.hstack([X_continuous.values, X_categorical.values, X_binary.values])
print("🎉 Base feature space extraction successful.")
del X_raw, X_continuous, X_categorical, X_binary; gc.collect()


# =========================================================================
# === HYBRID CONFIGURATION LAYER: MI + PCA + KMEANSSMOTE ===
# =========================================================================
print("\n=== Executing Configuration Pipeline (MI + PCA + KMeansSMOTE) ===")

# Step 1: Mutual Information Feature Selection
print("-> Step 1: Running Mutual Information Filtering...")
sample_size = min(50000, len(X_processed))
idx_sample = np.random.choice(len(X_processed), sample_size, replace=False)
mi_scores = mutual_info_classif(X_processed[idx_sample], y_all[idx_sample], random_state=42)
top_feature_indices = np.argsort(mi_scores)[-30:]  # Retain top 30 features
X_mi = X_processed[:, top_feature_indices]

# Step 2: Principal Component Analysis (PCA)
print("-> Step 2: Running Principal Component Analysis...")
pca = PCA(n_components=15, random_state=42)
X_pca = pca.fit_transform(X_mi)

# Step 3: KMeansSMOTE Balanced Sampling Vector
print("-> Step 3: Running Memory-Stabilized KMeansSMOTE Resampling...")
class_counts = np.bincount(y_all)
under_strategy = {i: min(count, 15000) for i, count in enumerate(class_counts)}
rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
X_rus, y_rus = rus.fit_resample(X_pca, y_all)

kms = KMeansSMOTE(cluster_balance_threshold=0.0, k_neighbors=2, random_state=42, n_jobs=1)
X_balanced, y_balanced = kms.fit_resample(X_rus, y_rus)
print(f"🎉 Resampling complete! Balanced Shape: {X_balanced.shape}")

del X_processed, X_mi, X_pca, X_rus; gc.collect()


# =========================================================================
# === PHASE 11: DEEP NEURAL NETWORK MODEL DEFINITIONS ===
# =========================================================================
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.network(x)

# =========================================================================
# === RUNNING STRATIFIED 5-FOLD VERIFICATION ===
# =========================================================================
print("\n=== Phase 11: Running Stratified Cross-Validation (DNN) ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced, y_balanced), 1):
    X_tr, y_tr = X_balanced[train_idx], y_balanced[train_idx]
    X_val, y_val = X_balanced[val_idx], y_balanced[val_idx]
    
    # Scale inside fold loops to protect verification clean states
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_val)
    
    # Convert to execution tensors
    X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)
    
    model = DeepNeuralNetwork(input_dim=X_balanced.shape[1], output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    model.train()
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(X_va_t.to(device)), dim=1).cpu().numpy()
        
    y_val_bin = (y_val != normal_class_idx).astype(int)
    val_preds_bin = (val_preds != normal_class_idx).astype(int)
    
    fold_metrics.append([
        fold,
        accuracy_score(y_val_bin, val_preds_bin),
        f1_score(y_val_bin, val_preds_bin, average='binary'),
        accuracy_score(y_val, val_preds),
        f1_score(y_val, val_preds, average='macro'),
        f1_score(y_val, val_preds, average='weighted')
    ])
    print(f"🧠 Fold {fold} / 5 Processing Complete.")

# =========================================================================
# === OUTPUT SUMMARY MATRIX DISPLAY ===
# =========================================================================
print("\n=== Cross-Validation Matrix (MI + PCA + KMeansSMOTE + DNN Pipeline) ===")
df_res = pd.DataFrame(fold_metrics, columns=['Fold', 'Binary Acc', 'Binary F1', 'Multi-Acc', 'Multi-F1 (Macro)', 'Weighted F1'])
print(df_res.to_string(index=False))

print("-" * 85)
print(f"Mean Average |  {df_res['Binary Acc'].mean():.6f}  |  {df_res['Binary F1'].mean():.6f}  |  {df_res['Multi-Acc'].mean():.6f}  |  {df_res['Multi-F1 (Macro)'].mean():.6f}  |  {df_res['Weighted F1'].mean():.6f}")
print("-" * 85)