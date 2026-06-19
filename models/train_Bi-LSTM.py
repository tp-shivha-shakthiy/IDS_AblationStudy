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

# Safe automated installation/import layer for external machine learning utilities
try:
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("Installing missing imbalanced-learn packages...")
    os.system('pip install imbalanced-learn')
    from imblearn.over_sampling import KMeansSMOTE
    from imblearn.under_sampling import RandomUnderSampler

try:
    import xgboost as xgb
except ImportError:
    print("Installing missing xgboost package...")
    os.system('pip install xgboost')
    import xgboost as xgb

# Set deterministic seeds for deep learning and boosting reproducibility
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

# Compute sample weights to represent Class Weighting inside the loss layers
class_counts = np.bincount(y_all)
total_samples = len(y_all)
computed_weights = total_samples / (num_classes * class_counts)
sample_weights_all = computed_weights[y_all]

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
# === PIPELINE GENERATOR 1: MI + PCA + KMEANSSMOTE CONFIGURATION ===
# =========================================================================
print("\n=== Processing Dataset Line 1 (MI + PCA + KMeansSMOTE) ===")
sample_size = min(50000, len(X_processed))
idx_sample = np.random.choice(len(X_processed), sample_size, replace=False)
mi_scores = mutual_info_classif(X_processed[idx_sample], y_all[idx_sample], random_state=42)
top_30_indices = np.argsort(mi_scores)[-30:]
X_mi = X_processed[:, top_30_indices]

pca = PCA(n_components=15, random_state=42)
X_pca = pca.fit_transform(X_mi)

counts_1 = np.bincount(y_all)
under_strat_1 = {i: min(c, 15000) for i, c in enumerate(counts_1)}
X_rus_1, y_rus_1 = RandomUnderSampler(sampling_strategy=under_strat_1, random_state=42).fit_resample(X_pca, y_all)
X_bal_legacy, y_bal_legacy = KMeansSMOTE(cluster_balance_threshold=0.0, k_neighbors=2, random_state=42, n_jobs=1).fit_resample(X_rus_1, y_rus_1)
print(f"🎉 Resampling 1 Complete. Shape: {X_bal_legacy.shape}")

del X_mi, X_pca, X_rus_1, y_rus_1; gc.collect()


# =========================================================================
# === PIPELINE GENERATOR 2: LOG + RAW SCALING + KMEANSSMOTE CONFIGURATION ===
# =========================================================================
print("\n=== Processing Dataset Line 2 (Log + Raw Scaling + KMeansSMOTE) ===")
counts_2 = np.bincount(y_all)
# Lean downsampling strategy to keep the high-fidelity features completely memory-safe on CPU
under_strat_2 = {i: min(c, 8000) for i, c in enumerate(counts_2)}
X_rus_2, y_rus_2 = RandomUnderSampler(sampling_strategy=under_strat_2, random_state=42).fit_resample(X_processed, y_all)
X_bal_raw_smote, y_bal_raw_smote = KMeansSMOTE(cluster_balance_threshold=0.0, k_neighbors=2, random_state=42, n_jobs=1).fit_resample(X_rus_2, y_rus_2)
print(f"🎉 Resampling 2 Complete. Shape: {X_bal_raw_smote.shape}")

del X_rus_2, y_rus_2; gc.collect()


# =========================================================================
# === MODEL DEFINITIONS: WEIGHTED BI-LSTM ===
# =========================================================================
class WeightedBidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightedBidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # Format to (Batch, Sequence Length = 1, Features)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# =========================================================================
# === EVALUATION LOOP: PIPELINE 1 (BI-LSTM WEIGHTED) ===
# =========================================================================
print("\n=== Running Stratified 5-Fold Evaluation: Legacy Weighted Bi-LSTM ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lstm_metrics = []

# Generate class weights specifically for the resampled legacy target vector space
resampled_counts = np.bincount(y_bal_legacy)
legacy_weights = len(y_bal_legacy) / (num_classes * resampled_counts)
legacy_weights_tensor = torch.tensor(legacy_weights, dtype=torch.float32).to(device)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_bal_legacy, y_bal_legacy), 1):
    X_tr, y_tr = X_bal_legacy[train_idx], y_bal_legacy[train_idx]
    X_val, y_val = X_bal_legacy[val_idx], y_bal_legacy[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_val)
    
    loader = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=512, shuffle=True)
    model = WeightedBidirectionalLSTM(input_dim=X_bal_legacy.shape[1], hidden_dim=32, output_dim=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=legacy_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(5):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X_va_s, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        
    y_val_bin = (y_val != normal_class_idx).astype(int)
    preds_bin = (preds != normal_class_idx).astype(int)
    
    lstm_metrics.append([accuracy_score(y_val_bin, preds_bin), f1_score(y_val_bin, preds_bin, average='binary'), accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), f1_score(y_val, preds, average='weighted')])
    print(f" Fold {fold} Processing Complete.")

df_lstm = pd.DataFrame(lstm_metrics, columns=['Binary Acc', 'Binary F1', 'Multi-Acc', 'Multi-F1 (Macro)', 'Weighted F1'])


# =========================================================================
# === EVALUATION LOOP: PIPELINE 2 (XGBOOST + KMEANSSMOTE) ===
# =========================================================================
print("\n=== Running Stratified 5-Fold Evaluation: Log Scaling + KMeansSMOTE + XGBoost ===")
xgb_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_bal_raw_smote, y_bal_raw_smote), 1):
    X_tr, y_tr = X_bal_raw_smote[train_idx], y_bal_raw_smote[train_idx]
    X_val, y_val = X_bal_raw_smote[val_idx], y_bal_raw_smote[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_val)
    
    # Configure high-performance multi-class tree boosting on CPU
    xgb_clf = xgb.XGBClassifier(n_estimators=40, max_depth=6, learning_rate=0.1, objective='multi:softprob', num_class=num_classes, tree_method='hist', n_jobs=-1, random_state=42)
    xgb_clf.fit(X_tr_s, y_tr)
    
    preds = xgb_clf.predict(X_va_s)
    
    y_val_bin = (y_val != normal_class_idx).astype(int)
    preds_bin = (preds != normal_class_idx).astype(int)
    
    xgb_metrics.append([accuracy_score(y_val_bin, preds_bin), f1_score(y_val_bin, preds_bin, average='binary'), accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), f1_score(y_val, preds, average='weighted')])
    print(f" Fold {fold} Processing Complete.")

df_xgb = pd.DataFrame(xgb_metrics, columns=['Binary Acc', 'Binary F1', 'Multi-Acc', 'Multi-F1 (Macro)', 'Weighted F1'])


# =========================================================================
# === FINAL METRIC CONSOLIDATION DISPLAY ===
# =========================================================================
print("\n" + "="*50 + "\nFINAL EXECUTION RESULTS MATRIX\n" + "="*50)
print("\n1. MI + PCA + Log + KMeansSMOTE + Weighted Bi-LSTM Performance Summary:")
print(f"Mean Binary Acc : {df_lstm['Binary Acc'].mean():.6f}")
print(f"Mean Binary F1  : {df_lstm['Binary F1'].mean():.6f}")
print(f"Mean Multi Acc  : {df_lstm['Multi-Acc'].mean():.6f}")
print(f"Mean Macro F1   : {df_lstm['Multi-F1 (Macro)'].mean():.6f}")
print(f"Mean Weighted F1: {df_lstm['Weighted F1'].mean():.6f}")

print("\n2. Log + Raw Scaling + KMeansSMOTE + XGBoost Performance Summary:")
print(f"Mean Binary Acc : {df_xgb['Binary Acc'].mean():.6f}")
print(f"Mean Binary F1  : {df_xgb['Binary F1'].mean():.6f}")
print(f"Mean Multi Acc  : {df_xgb['Multi-Acc'].mean():.6f}")
print(f"Mean Macro F1   : {df_xgb['Multi-F1 (Macro)'].mean():.6f}")
print(f"Mean Weighted F1: {df_xgb['Weighted F1'].mean():.6f}")