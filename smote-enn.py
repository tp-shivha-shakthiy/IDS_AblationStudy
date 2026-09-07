import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from imblearn.combine import SMOTEENN
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

print("Starting multi-class pipeline...")

# Load & merge all parts
paths = ['data/UNSW-NB15_1.csv', 'data/UNSW-NB15_2.csv', 'data/UNSW-NB15_3.csv', 'data/UNSW-NB15_4.csv']
column_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
    'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
    'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

data = pd.concat([pd.read_csv(p, header=None) for p in paths], ignore_index=True)
data.columns = column_names
print("\u2705 Data loaded.")

# Drop leakage columns
leakage_cols = ['srcip', 'dstip', 'sport', 'dsport', 'stime', 'ltime']
data.drop(columns=leakage_cols, inplace=True)

# Drop rows with missing attack category
data = data[~data['attack_cat'].isna()]

# Handle missing/infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in data.columns:
    if data[col].dtype != 'object':
        data[col].fillna(data[col].mean(), inplace=True)
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical features
for col in data.select_dtypes(include='object').columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Feature & target separation
X = data.drop(columns=['attack_cat', 'label'])
y = data['attack_cat'].values
class_names = np.unique(y)
n_classes = len(class_names)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pearson feature selection
def select_features_by_pearson(X, y, threshold=0.15):
    selected = []
    for i in range(X.shape[1]):
        try:
            corr, _ = pearsonr(X[:, i], y)
            if abs(corr) > threshold:
                selected.append(i)
        except:
            continue
    return selected

selected_idxs = select_features_by_pearson(X_scaled, y)
X_selected = X_scaled[:, selected_idxs] if selected_idxs else X_scaled
print(f"\u2705 Selected {len(selected_idxs)} features using Pearson correlation.")

# Binarize target for ROC
y_bin = label_binarize(y, classes=class_names)

# Models
models = {
    "XGBoost": OneVsRestClassifier(XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                                 use_label_encoder=False, eval_metric='mlogloss')),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Decision Tree": OneVsRestClassifier(DecisionTreeClassifier(random_state=42)),
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    "KNN": OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print("\u2705 Starting multi-class cross-validation...")

# Train & evaluate
for name, model in models.items():
    print(f"\n\U0001f50d Training model: {name}")
    all_acc, all_prec, all_rec, all_f1_macro, all_f1_weighted, all_auc = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_selected, y), 1):
        print(f"\u27a1 Fold {fold}")
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_bin, y_test_bin = y_bin[train_idx], y_bin[test_idx]

        # Handle imbalance using SMOTEENN
        smt = SMOTEENN(random_state=42)
        X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
        y_train_bin_res = label_binarize(y_train_res, classes=class_names)

        # Train model
        model.fit(X_train_res, y_train_bin_res)
        y_pred_bin = model.predict(X_test)
        y_pred = y_pred_bin.argmax(axis=1)
        y_prob = model.predict_proba(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

        all_acc.append(acc)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1_macro.append(f1_macro)
        all_f1_weighted.append(f1_weighted)
        all_auc.append(auc)

        print(f"\u2705 Fold {fold} - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, "
              f"F1(Macro): {f1_macro:.4f}, F1(Weighted): {f1_weighted:.4f}, AUC: {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f'{name} - Fold {fold} Confusion Matrix')
        plt.tight_layout()
        plt.show()

        # ROC Curve (only for Fold 1)
        if fold == 1:
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tprs.append(tpr_interp)
                plt.plot(fpr, tpr, label=f'{class_names[i]} (Class {i})')

    print(f"\n\U0001f4ca Avg Results for {name}:")
    print(f"Accuracy:       {np.mean(all_acc):.4f}")
    print(f"Precision:      {np.mean(all_prec):.4f}")
    print(f"Recall:         {np.mean(all_rec):.4f}")
    print(f"F1 Macro:       {np.mean(all_f1_macro):.4f}")
    print(f"F1 Weighted:    {np.mean(all_f1_weighted):.4f}")
    print(f"ROC AUC (OVR):  {np.mean(all_auc):.4f}")

    # Plot mean ROC curve
    if tprs:
        avg_tpr = np.mean(tprs, axis=0)
        plt.plot(mean_fpr, avg_tpr, label=f'{name} (macro AUC={np.mean(all_auc):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curves')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

print(" All models trained and evaluated.")