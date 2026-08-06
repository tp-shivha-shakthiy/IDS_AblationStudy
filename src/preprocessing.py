"""
preprocessing.py
================
Phase 3 — Preprocessing Layer

Handles:
  - Loading raw UNSW-NB15 CSV files
  - Target column cleaning and class mapping
  - Label encoding (categorical features + target)
  - Log1p normalisation (scale stabilisation)

Returns:
  X_processed  : float32 numpy array  (log-normalised feature matrix)
  y_multi      : int numpy array       (encoded multi-class labels)
  le           : fitted LabelEncoder   (target, shared across all modules)
"""

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COL_NAMES = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
    'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'sintpkt', 'dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_src_ltm',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm_d',
    'ct_srv_dst', 'ct_state_ttl', 'ct_src_user_ltm', 'ct_src_zone_ltm',
    'ct_dst_host_ltm', 'ct_srv_src', 'attack_cat', 'label',
]

CATEGORY_MAPPING = {
    'normal': 'Normal',
    'fuzzers': 'Fuzzers',
    'analysis': 'Analysis',
    'backdoor': 'Backdoor',
    'dos': 'DoS',
    'exploits': 'Exploits',
    'generic': 'Generic',
    'reconnaissance': 'Reconnaissance',
    'shellcode': 'Shellcode',
    'worms': 'Worms',
}

DROP_COLS = ['id', 'label', 'stime', 'ltime', 'srcip', 'dstip']
TARGET_COL = 'attack_cat'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess(data_dir: str = "data/raw") -> tuple:
    """
    Load all UNSW-NB15 CSV files from *data_dir*, clean and preprocess them.

    Parameters
    ----------
    data_dir : str
        Path containing UNSW-NB15_1.csv … UNSW-NB15_4.csv

    Returns
    -------
    X_processed : np.ndarray  shape (N, F)   float32, log-normalised
    y_multi     : np.ndarray  shape (N,)     int, encoded attack categories
    le          : LabelEncoder               fitted on target column
    """
    print("=== Phase 3: Commencing Preprocessing Layer ===")

    # ------------------------------------------------------------------
    # 1. Load raw files
    # ------------------------------------------------------------------
    df_list = []
    for i in range(1, 5):
        fname = f"{data_dir}/UNSW-NB15_{i}.csv"
        try:
            print(f"  Loading {fname} ...")
            df_temp = pd.read_csv(fname, header=None, low_memory=False)

            if df_temp.shape[1] == 49:
                df_temp.columns = COL_NAMES[:47] + ['attack_cat', 'label']
            else:
                df_temp.columns = COL_NAMES[:df_temp.shape[1]]

            df_list.append(df_temp)
        except FileNotFoundError:
            print(f"  Warning: {fname} not found - skipping.")

    if not df_list:
        raise FileNotFoundError(
            f"No UNSW-NB15 CSV files found in '{data_dir}'. "
            "Download the dataset and place the four CSV files there."
        )

    df = pd.concat(df_list, ignore_index=True)
    print(f"  Data ingestion complete. Combined shape: {df.shape}")
    del df_list; gc.collect()

    # ------------------------------------------------------------------
    # 2. Target cleaning & mapping
    # ------------------------------------------------------------------
    df[TARGET_COL] = (
        df[TARGET_COL]
        .fillna('Normal')
        .astype(str)
        .str.strip()
        .str.lower()
        .map(CATEGORY_MAPPING)
        .fillna('Normal')
    )

    le = LabelEncoder()
    y_multi = le.fit_transform(df[TARGET_COL])
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 3. Drop metadata columns
    # ------------------------------------------------------------------
    drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X_raw = df.drop(columns=drop)
    del df; gc.collect()

    # ------------------------------------------------------------------
    # 4. Encode remaining categorical columns
    # ------------------------------------------------------------------
    cat_cols = X_raw.select_dtypes(include=['object']).columns.tolist()
    print(f"  Encoding categorical features: {cat_cols}")
    for col in cat_cols:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    # ------------------------------------------------------------------
    # 5. Log1p normalisation  (clip → log1p → fillna)
    # ------------------------------------------------------------------
    print("  Applying log1p normalisation ...")
    X_processed = (
        np.log1p(X_raw.clip(lower=0))
        .fillna(0)
        .astype('float32')
        .values
    )
    del X_raw; gc.collect()

    print(f"  Preprocessing complete. Feature matrix shape: {X_processed.shape}")
    return X_processed, y_multi, le
