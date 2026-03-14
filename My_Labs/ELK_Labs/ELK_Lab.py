import kagglehub
import elasticsearch
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import RandomizedSearchCV,train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report,f1_score,confusion_matrix
import logging
import numpy as np
import warnings
import json
import os
import logging
os.makedirs("logs", exist_ok=True)
os.makedirs("pipeline", exist_ok=True)
os.makedirs("metricbeat", exist_ok=True)
os.makedirs("filebeat", exist_ok=True)
os.makedirs("datasets", exist_ok=True)
logging.basicConfig(
    filename="logs/fraud_model.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
warnings.filterwarnings("ignore")

os.environ["KAGGLEHUB_CACHE"]=''
path = kagglehub.dataset_download("algozee/financial-transaction-fraud-dataset")
csv_folder = Path(f'./{path}')
param_grids = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C":            [0.01, 0.1, 1, 10, 100],
            "penalty":      ["l1", "l2"],
            "solver":       ["liblinear", "saga"],
            "class_weight": [None, "balanced"]
        }
    },
    "LinearSVC": {
        "model": CalibratedClassifierCV(LinearSVC()),
        "params": {
            "estimator__C":            [0.01, 0.1, 1, 10, 100],
            "estimator__penalty":      ["l1", "l2"],
            "estimator__loss":         ["hinge", "squared_hinge"],
            "estimator__class_weight": [None, "balanced"]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "max_depth":         [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "criterion":         ["gini", "entropy"],
            "class_weight":      [None, "balanced"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators":      [100, 200],        
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf":  [1, 2],
            "max_features":      ["sqrt", "log2"],
            "class_weight":      [None, "balanced"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan", "minkowski"],
            "p":           [1, 2]
        }
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        "params": {
            "n_estimators":     [100, 200],         
            "learning_rate":    [0.01, 0.05, 0.1, 0.3],
            "max_depth":        [3, 5, 7],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "scale_pos_weight": [15, 20, 25]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(n_iter_no_change=5, validation_fraction=0.1), 
        "params": {
            "n_estimators":  [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.8, 1.0]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(),
        "params": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(verbose=-1),
        "params": {
            "n_estimators":     [100, 200],         
            "learning_rate":    [0.01, 0.05, 0.1],
            "max_depth":        [-1, 5, 10],
            "num_leaves":       [31, 63, 127],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "class_weight":     [None, "balanced"]
        }
    }
}
merged_df = pd.concat(pd.read_csv(p) for p in csv_folder.glob('*.csv'))
merged_df.head()
merged_df.drop_duplicates(inplace=True)
merged_df.dropna(inplace=True)
cols_to_drop=[
    'Transaction_ID', 'Customer_ID', 'Merchant_ID', 'Device_ID', 'IP_Address', 'Transaction_Location', 'Customer_Home_Location',
    "Transaction_Time","Transaction_Date"
]
merged_df["Transaction_Date"]=pd.to_datetime(merged_df["Transaction_Date"])
merged_df["day_of_week"]=merged_df["Transaction_Date"].dt.dayofweek
merged_df["month"]=merged_df["Transaction_Date"].dt.month
merged_df["is_weekend"]=merged_df["Transaction_Date"].dt.dayofweek.isin([5, 6]).astype(int)
merged_df["Transaction_Time"] = pd.to_datetime(merged_df["Transaction_Time"], format="%H:%M")
merged_df["hour"] = merged_df["Transaction_Time"].dt.hour
merged_df.drop(columns=cols_to_drop,inplace=True)
merged_df["Is_International_Transaction"]=merged_df["Is_International_Transaction"].map({"Yes": 1, "No": 0})
merged_df["Is_New_Merchant"]=merged_df["Is_New_Merchant"].map({"Yes": 1, "No": 0})
merged_df["Unusual_Time_Transaction"]=merged_df["Unusual_Time_Transaction"].map({"Yes": 1, "No": 0})
merged_df["Fraud_Label"]=merged_df["Fraud_Label"].map({"Fraud": 1, "Normal": 0})
int_cols=["Daily_Transaction_Count", "Weekly_Transaction_Count",
            "Failed_Transaction_Count", "Previous_Fraud_Count"]
merged_df[int_cols]=merged_df[int_cols].astype(int)
for col in ["Transaction_Type", "Merchant_Category", "Card_Type"]:
    merged_df[col] = LabelEncoder().fit_transform(merged_df[col])
    X = merged_df.drop(columns=["Fraud_Label"])
y = merged_df["Fraud_Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

for name, cfg in param_grids.items():
    print(f"Running {name}...")
    gs = RandomizedSearchCV(
        estimator=cfg["model"],
        param_distributions=cfg["params"],
        n_iter=30,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        pre_dispatch="2*n_jobs",
        verbose=1,
    )
    gs.fit(X_train, y_train)
    results[name] = {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score":  gs.best_score_,
        "report":      classification_report(y_test, gs.predict(X_test)),
    }
results_df=pd.DataFrame(results)
best_name = max(results, key=lambda x: results[x]["best_score"])
best_name = max(results, key=lambda x: results[x]["best_score"])
best_model = results[best_name]["best_estimator"]
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df["fraud_prediction"] = y_pred
X_test_df["fraud_probability"] = y_prob
X_test_df["actual_label"] = y_test.values
f1 = f1_score(y_test, y_pred, average='weighted')  
conf_matrix = confusion_matrix(y_test, y_pred)
tp = np.diag(conf_matrix)
tn = np.sum(conf_matrix) - (np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - tp)
fp = np.sum(conf_matrix, axis=0) - tp
fn = np.sum(conf_matrix, axis=1) - tp
fp_rate = fp / (fp + tn)
fn_rate = fn / (fn + tp)
logging.info(f"F1 Score: {f1:.2f}")
logging.info(f"True Negative: {tn}")
logging.info(f"False Positive Rate: {fp_rate}")
logging.info(f"False Negative Rate: {fn_rate}")
logging.info(f"True Positive: {tp}")
logging.info(f"False Positive Rate: {fp_rate:}")
logging.info(f"False Negative Rate: {fn_rate:}")
logging.info(f"Feature Importances: {best_model.feature_importances_}")
X_test_df["timestamp"] = pd.Timestamp.now().isoformat()
records = X_test_df.to_dict(orient="records")
with open("pipeline/fraud_predictions.json", "w") as f:
    for record in records:
        json.dump(record, f)
        f.write("\n")        
print(f"Exported {len(records)} records")
