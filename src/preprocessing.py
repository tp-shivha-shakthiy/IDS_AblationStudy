"""
preprocessing.py
================
Phase 3 — Preprocessing Layer

Handles:
  - Loading raw UNSW-NB15 CSV files
  - Target column cleaning and class mapping
   - Target label encoding and deterministic feature cleaning
  - Log1p normalisation (scale stabilisation)

Returns:
   X_features   : DataFrame (unencoded feature matrix)
  y_multi      : int numpy array       (encoded multi-class labels)
  le           : fitted LabelEncoder   (target, shared across all modules)
"""

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
    'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime',
    'sintpkt', 'dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label',
]

SCHEMA_VERSION = "unsw_nb15_raw_49_v1"
EXPECTED_RAW_COLUMNS = 49
CATEGORICAL_COLUMNS = ('proto', 'state', 'service')

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

# Fixed, dataset-defined target vocabulary.  This avoids fitting a target
# encoder on labels that will later form the locked test set.
TARGET_CLASSES = (
    'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
    'Normal', 'Reconnaissance', 'Shellcode', 'Worms',
)

DROP_COLS = ['id', 'label', 'stime', 'ltime', 'srcip', 'dstip']
TARGET_COL = 'attack_cat'


def feature_columns() -> list[str]:
    """Return the canonical ordered model feature names after intentional drops."""
    return [name for name in RAW_COLUMNS if name not in {*DROP_COLS, TARGET_COL}]


FEATURE_COLUMNS = feature_columns()
NUMERIC_COLUMNS = tuple(name for name in FEATURE_COLUMNS if name not in CATEGORICAL_COLUMNS)

if len(RAW_COLUMNS) != EXPECTED_RAW_COLUMNS or len(set(RAW_COLUMNS)) != EXPECTED_RAW_COLUMNS:
    raise RuntimeError("UNSW-NB15 raw schema must contain 49 unique columns.")
if not set(CATEGORICAL_COLUMNS).issubset(FEATURE_COLUMNS):
    raise RuntimeError("Categorical schema fields must be model features.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_prepare(data_dir: str = "data/raw") -> tuple:
    """
    Load all UNSW-NB15 CSV files and perform only pre-split-safe preparation.

    Categorical feature encoding is deliberately excluded here.  It is a
    learned operation and must be fitted after the final holdout split.

    Parameters
    ----------
    data_dir : str
        Path containing UNSW-NB15_1.csv … UNSW-NB15_4.csv

    Returns
    -------
    X_features  : pd.DataFrame shape (N, F)  unencoded feature matrix
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

            if df_temp.shape[1] != EXPECTED_RAW_COLUMNS:
                raise ValueError(
                    f"{fname} has {df_temp.shape[1]} columns; expected "
                    f"{EXPECTED_RAW_COLUMNS} for schema {SCHEMA_VERSION}."
                )
            df_temp.columns = RAW_COLUMNS

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
    le.classes_ = np.asarray(TARGET_CLASSES, dtype=object)
    y_multi = le.transform(df[TARGET_COL])
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 3. Drop metadata columns
    # ------------------------------------------------------------------
    drop = [c for c in DROP_COLS if c in df.columns] + [TARGET_COL]
    X_raw = df.drop(columns=drop)
    del df; gc.collect()

    # ------------------------------------------------------------------
    # 4. Keep remaining categorical columns unencoded.  Their encoder is fit
    # after the holdout split, using training data only.
    # ------------------------------------------------------------------
    if list(X_raw.columns) != FEATURE_COLUMNS:
        raise RuntimeError("Post-drop UNSW feature order does not match canonical schema.")
    cat_cols = list(CATEGORICAL_COLUMNS)
    print(f"  Categorical features deferred until post-split encoding: {cat_cols}")
    print(f"  Pre-split preparation complete. Feature matrix shape: {X_raw.shape}")
    return X_raw, y_multi, le


def fit_categorical_encoder(X_train: pd.DataFrame):
    """Fit an unknown-safe categorical encoder on training features only."""
    # pandas 3.x stores strings as 'str' (StringDtype) or legacy 'object'.
    cat_cols = [name for name in CATEGORICAL_COLUMNS if name in X_train.columns]

    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1,
        dtype=np.float32,
    )
    encoder.fit(X_train[cat_cols].astype(str))
    return encoder


def transform_features(
    X: pd.DataFrame,
    categorical_encoder,
    preserve_categorical_codes: bool = False,
) -> np.ndarray:
    """Encode with a train-fitted encoder, then apply deterministic transforms."""
    X_out = X.copy()
    if categorical_encoder is not None:
        cat_cols = list(categorical_encoder.feature_names_in_)
        encoded = categorical_encoder.transform(X_out[cat_cols].astype(str))
        # Replace the categorical columns wholesale with the encoder's numeric
        # output instead of assigning floats into str/StringDtype columns.
        enc_block = pd.DataFrame(
            encoded, index=X_out.index, columns=cat_cols,
        )
        X_out = pd.concat(
            [X_out.drop(columns=cat_cols), enc_block], axis=1,
        )[X_out.columns]

    X_out = X_out.apply(pd.to_numeric, errors='coerce')
    if preserve_categorical_codes and categorical_encoder is not None:
        # MI's discrete branch requires the ordinal category codes themselves,
        # not their log-transformed continuous values.
        numeric_cols = X_out.columns.difference(categorical_encoder.feature_names_in_)
        X_out[numeric_cols] = np.log1p(X_out[numeric_cols].clip(lower=0))
        X_values = X_out.fillna(0).astype('float32').values
    else:
        X_values = (
            np.log1p(X_out.clip(lower=0))
            .fillna(0)
            .astype('float32')
            .values
        )
    if list(X.columns) == FEATURE_COLUMNS and X_values.shape[1] != len(FEATURE_COLUMNS):
        raise RuntimeError("Transformed canonical feature width does not match schema metadata.")
    return X_values


def feature_type_mask(feature_names: list[str] | None = None) -> np.ndarray:
    """Boolean MI mask aligned to the supplied transformed feature order."""
    if feature_names is None:
        feature_names = FEATURE_COLUMNS
    return np.asarray([name in CATEGORICAL_COLUMNS for name in feature_names], dtype=bool)


def feature_metadata() -> dict:
    """Return schema-aligned metadata required by MI and raw inference."""
    mask = feature_type_mask()
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_columns": list(RAW_COLUMNS),
        "feature_names": list(FEATURE_COLUMNS),
        "dropped_columns": list(DROP_COLS) + [TARGET_COL],
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "discrete_mask": mask,
        "discrete_indices": np.flatnonzero(mask).tolist(),
        "numeric_indices": np.flatnonzero(~mask).tolist(),
        "unknown_category_policy": "ordinal_unknown_minus_one_then_log1p_zero",
    }
