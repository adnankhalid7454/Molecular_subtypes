import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import joblib, os
from catboost import CatBoostClassifier


def load_bundle(metadata_path: str):
    meta = joblib.load(metadata_path)

    scaler = joblib.load(meta["scaler_path"])
    feature_names = meta["feature_names"]
    classes = np.array(list(map(str, meta["classes"])))

    model = joblib.load(meta["model_pkl_path"])    #Load Model

    if isinstance(model, CatBoostClassifier) and meta.get("cbm_path"):
        try:
            model_native = CatBoostClassifier()
            model_native.load_model(meta["cbm_path"], format="cbm")
            model = model_native
        except Exception:
            pass

    return model, scaler, feature_names, classes, meta


def load_frame_auto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file type, use CSV or Excel")


def evaluate_saved_model(metadata_path: str, dataset_path: str, label_column: str = "Subtypes"):

    # ---- Load metadata + model ----
    meta = joblib.load(metadata_path)
    scaler = joblib.load(meta["scaler_path"])
    feature_names = meta["feature_names"]
    classes = np.array(meta["classes"], dtype=str)  # enforce consistent type

    model = joblib.load(meta["model_pkl_path"])
    if isinstance(model, CatBoostClassifier) and meta.get("cbm_path"):
        try:
            model_native = CatBoostClassifier()
            model_native.load_model(meta["cbm_path"], format="cbm")
            model = model_native
        except Exception:
            pass

    # ---- Loading new dataset ----
    df = pd.read_excel(dataset_path) if dataset_path.endswith("xlsx") else pd.read_csv(dataset_path)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")

    X = df[feature_names].copy()
    X_scaled = scaler.transform(X)

    has_labels = label_column in df.columns
    if has_labels:
        y_true = df[label_column].astype(str).values  # cast labels to str
    else:
        y_true = None

    # ---- Predictions ----
    y_pred = model.predict(X_scaled).astype(str)
    y_proba = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

    if has_labels:
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")

        if y_proba is not None:
            y_true_bin = label_binarize(y_true, classes=classes)
            auc = roc_auc_score(y_true_bin, y_proba[:, np.arange(len(classes))], 
                                average="weighted", multi_class="ovr")
            print(f"Weighted AUC: {auc:.4f}")
        else:
            auc = np.nan
            print("Weighted AUC not available")

        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {meta['model_name']}")
        plt.tight_layout()
        plt.show()

        return {"accuracy": acc, "auc_weighted": auc}

    else:
        print("No labels found, saving predictions...")
        out = df.copy()
        out[f"{meta['model_name']}_Pred"] = y_pred
        if y_proba is not None:
            proba_df = pd.DataFrame(y_proba, columns=[f"P({c})" for c in classes])
            out = pd.concat([out.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        out.to_excel("predictions_no_labels.xlsx", index=False)
        return {"predictions_path": "predictions_no_labels.xlsx"}



# metadata_path = "saved_models/metadata_CatBoost_20250827_024029.joblib"
# dataset_path  = "test_feature.xlsx"
# evaluate_saved_model(metadata_path, dataset_path, label_column="Subtypes")
