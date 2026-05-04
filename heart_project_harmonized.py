import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# -----------------------------
# data loading
# -----------------------------
def load_uci_data():
    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    print("Loaded dataset from UCI.")
    return X, y


def load_kaggle_data(file_path="heart_kaggle.csv", drop_dup=True):
    df = pd.read_csv(file_path)
    print(f"Loaded Kaggle dataset from {file_path}.")
    print("Original Kaggle shape:", df.shape)

    dup_count = df.duplicated().sum()
    print("Duplicate rows in Kaggle:", dup_count)

    if drop_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Kaggle shape after deduplication:", df.shape)

    return df


# -----------------------------
# plotting
# -----------------------------
def plot_class_distribution(y, file_name, title):
    counts = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(5, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


def plot_top_features(df, value_col, file_name, title, top_n=8):
    plot_df = df.head(top_n).copy()

    plt.figure(figsize=(7, 4))
    plt.barh(plot_df["feature"][::-1], plot_df[value_col][::-1])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


# -----------------------------
# cleaning and harmonization
# -----------------------------
def clean_features(X):
    X = X.copy()
    X.columns = [str(col).strip().lower() for col in X.columns]

    X = X.replace("?", np.nan)
    X = X.replace("NA", np.nan)
    X = X.replace("N/A", np.nan)
    X = X.replace("", np.nan)

    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except Exception:
            pass

    return X


def clean_target(y, binary_mode="uci"):
    y = pd.Series(y).copy()
    y = y.replace("?", np.nan)
    y = pd.to_numeric(y, errors="coerce")

    if y.isnull().any():
        raise ValueError("Target column contains missing or invalid values.")

    if binary_mode == "uci":
        # UCI original target: 0 means no disease, >0 means disease.
        y = (y > 0).astype(int)
    else:
        # Kaggle target is already 0/1 in the johnsmith88 dataset.
        y = y.astype(int)

    return y


def print_encoded_value_ranges(X, name):
    print(f"\nEncoded value check for {name}:")
    for col in ["cp", "restecg", "slope", "ca", "thal"]:
        if col in X.columns:
            vals = sorted(pd.Series(X[col]).dropna().unique().tolist())
            print(f"  {col}: {vals}")


def harmonize_kaggle_to_uci_encoding(X):
    """
    Convert Kaggle johnsmith88 heart.csv categorical encodings to match
    the older UCI Cleveland-style encodings used by ucimlrepo.

    Kaggle / cleaned Cleveland-style source:
      cp:    0 Typical, 1 Atypical, 2 Non-Anginal, 3 Asymptomatic
      slope: 0 Upsloping, 1 Flat, 2 Downsloping
      thal:  1 Normal, 2 Fixed Defect, 3 Reversible Defect, 0 anomalous
      ca:    0-3 valid, 4 anomalous

    UCI Cleveland-style encodings in this project:
      cp:    1 Typical, 2 Atypical, 3 Non-Anginal, 4 Asymptomatic
      slope: 1 Upsloping, 2 Flat, 3 Downsloping
      thal:  3 Normal, 6 Fixed Defect, 7 Reversible Defect
      ca:    0-3 valid, missing values imputed later

    restecg, sex, fbs, and exang already use compatible encodings.
    """
    X = X.copy()

    # chest pain type: 0-3 -> 1-4
    if "cp" in X.columns:
        vals = set(pd.Series(X["cp"]).dropna().unique())
        if vals.issubset({0, 1, 2, 3}):
            X["cp"] = X["cp"] + 1

    # slope: 0-2 -> 1-3
    if "slope" in X.columns:
        vals = set(pd.Series(X["slope"]).dropna().unique())
        if vals.issubset({0, 1, 2}):
            X["slope"] = X["slope"] + 1

    # thal: 1,2,3 -> 3,6,7; 0 is anomalous and treated as missing
    if "thal" in X.columns:
        X.loc[X["thal"] == 0, "thal"] = np.nan
        vals = set(pd.Series(X["thal"]).dropna().unique())
        if vals.issubset({1, 2, 3}):
            X["thal"] = X["thal"].map({1: 3, 2: 6, 3: 7})

    # ca: 0-3 valid; 4 is anomalous and treated as missing
    if "ca" in X.columns:
        X.loc[X["ca"] == 4, "ca"] = np.nan

    return X


def align_kaggle_columns(df, harmonize_to_uci=True):
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    possible_target_cols = ["target", "num", "heartdisease", "label", "output"]
    target_col = None
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError("Could not find target column in Kaggle dataset.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    rename_map = {
        "trest_bps": "trestbps",
        "restingbp": "trestbps",
        "resting_bp_s": "trestbps",
        "cholesterol": "chol",
        "maxheartrate": "thalach",
        "max_heart_rate": "thalach",
        "maxhr": "thalach",
        "exerciseangina": "exang",
        "exercise_angina": "exang",
        "st_slope": "slope",
        "chestpaintype": "cp",
        "chest_pain_type": "cp",
        "fastingbs": "fbs",
        "fasting_bs": "fbs",
        "restingecg": "restecg",
        "resting_ecg": "restecg"
    }

    X = X.rename(columns=rename_map)
    X = clean_features(X)

    expected_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    missing_cols = [c for c in expected_cols if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Kaggle dataset is missing expected columns: {missing_cols}")

    X = X[expected_cols].copy()

    print_encoded_value_ranges(X, "Kaggle before harmonization")
    if harmonize_to_uci:
        X = harmonize_kaggle_to_uci_encoding(X)
        print_encoded_value_ranges(X, "Kaggle after harmonization to UCI")

    return X, y


# -----------------------------
# exploration
# -----------------------------
def explore_data(X, y, name):
    print(f"\n===== Data Exploration: {name} =====")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nColumns:")
    print(list(X.columns))

    print("\nFirst 5 rows:")
    print(X.head())

    print("\nMissing values per column:")
    print(X.isnull().sum())

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nSummary statistics:")
    print(X.describe(include="all"))


# -----------------------------
# preprocessing
# -----------------------------
def build_preprocessor(X):
    numeric_cols = X.columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols)
    ])

    return preprocessor


# -----------------------------
# tuning
# -----------------------------
def tune_model(name, pipeline, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print(f"\n===== {name} Tuning =====")
    print("Best Params:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))

    return grid


# -----------------------------
# evaluation
# -----------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_prob)

    print(f"\n===== {name} Final Evaluation =====")
    print("Train Accuracy:", round(train_acc, 4))
    print("Test Accuracy:", round(test_acc, 4))
    print("F1-score:", round(test_f1, 4))
    print("Precision:", round(test_precision, 4))
    print("Recall:", round(test_recall, 4))
    print("ROC-AUC:", round(test_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)

    return {
        "name": name,
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "f1": test_f1,
        "precision": test_precision,
        "recall": test_recall,
        "auc": test_auc,
        "cm": cm,
        "prob": test_prob,
        "pred": test_pred
    }


def save_summary(results, file_name):
    summary = pd.DataFrame([
        {
            "Model": r["name"],
            "Train Accuracy": round(r["train_acc"], 4),
            "Test Accuracy": round(r["test_acc"], 4),
            "F1-score": round(r["f1"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "ROC-AUC": round(r["auc"], 4)
        }
        for r in results
    ])
    summary.to_csv(file_name, index=False)
    print(f"\nSaved {file_name}")
    print(summary)
    return summary


def plot_confusion(results, file_name):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(confusion_matrix=res["cm"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(res["name"])

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


def plot_roc(results, y_test, file_name, title="ROC Curve"):
    plt.figure(figsize=(6, 5))

    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["prob"])
        label = f'{res["name"]} (AUC={res["auc"]:.3f})'
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


def save_linear_features(best_model, file_name):
    clf = best_model.best_estimator_.named_steps["classifier"]
    preprocessor = best_model.best_estimator_.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(file_name, index=False)
    print(f"\nSaved {file_name}")
    print(coef_df.head(10))
    return coef_df


def save_tree_features(best_model, file_name):
    clf = best_model.best_estimator_.named_steps["classifier"]
    preprocessor = best_model.best_estimator_.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = clf.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(file_name, index=False)
    print(f"\nSaved {file_name}")
    print(imp_df.head(10))
    return imp_df


def save_rf_features(best_model, file_name):
    clf = best_model.best_estimator_.named_steps["classifier"]
    preprocessor = best_model.best_estimator_.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = clf.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(file_name, index=False)
    print(f"\nSaved {file_name}")
    print(imp_df.head(10))
    return imp_df


# -----------------------------
# main experiment on UCI
# -----------------------------
def run_main_experiment():
    X, y = load_uci_data()
    X = clean_features(X)
    y = clean_target(y, binary_mode="uci")

    explore_data(X, y, "UCI")
    plot_class_distribution(y, "uci_class_distribution.png", "UCI Class Distribution")

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            random_state=RANDOM_STATE
        ))
    ])

    tree_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    logistic_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10]
    }

    tree_param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 5]
    }

    rf_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    best_logistic = tune_model(
        "Logistic Regression",
        logistic_pipeline,
        logistic_param_grid,
        X_train,
        y_train
    )

    best_tree = tune_model(
        "Decision Tree",
        tree_pipeline,
        tree_param_grid,
        X_train,
        y_train
    )

    best_rf = tune_model(
        "Random Forest",
        rf_pipeline,
        rf_param_grid,
        X_train,
        y_train
    )

    logistic_result = evaluate_model(
        "Logistic Regression",
        best_logistic.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    tree_result = evaluate_model(
        "Decision Tree",
        best_tree.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    rf_result = evaluate_model(
        "Random Forest",
        best_rf.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    results = [logistic_result, tree_result, rf_result]

    save_summary(results, "uci_model_summary.csv")
    plot_confusion(results, "uci_confusion_matrices.png")
    plot_roc(results, y_test, "uci_roc_curve.png", title="UCI ROC Curve")

    logistic_features = save_linear_features(best_logistic, "uci_logistic_top_features.csv")
    tree_features = save_tree_features(best_tree, "uci_tree_top_features.csv")
    rf_features = save_rf_features(best_rf, "uci_rf_top_features.csv")

    plot_top_features(
        logistic_features,
        "abs_coefficient",
        "uci_logistic_top_features.png",
        "Top Logistic Regression Features on UCI"
    )

    plot_top_features(
        tree_features,
        "importance",
        "uci_tree_top_features.png",
        "Top Decision Tree Features on UCI"
    )

    plot_top_features(
        rf_features,
        "importance",
        "uci_rf_top_features.png",
        "Top Random Forest Features on UCI"
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_logistic": best_logistic,
        "best_tree": best_tree,
        "best_rf": best_rf,
        "results": results,
        "logistic_features": logistic_features,
        "tree_features": tree_features,
        "rf_features": rf_features
    }


# -----------------------------
# supplementary Kaggle experiment
# -----------------------------
def run_kaggle_supplementary_experiment():
    """
    This is NOT true external validation.
    It trains and tests on a deduplicated, harmonized Kaggle split.
    It is kept as a supplementary in-dataset experiment.
    """
    df = load_kaggle_data("heart_kaggle.csv", drop_dup=True)
    X, y = align_kaggle_columns(df, harmonize_to_uci=True)

    X = clean_features(X)
    y = clean_target(y, binary_mode="kaggle")

    explore_data(X, y, "Kaggle Supplementary")
    plot_class_distribution(y, "kaggle_supplementary_class_distribution.png", "Kaggle Supplementary Class Distribution")

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logistic_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            random_state=RANDOM_STATE
        ))
    ])

    tree_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    logistic_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10]
    }

    tree_param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 5]
    }

    rf_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    best_logistic = tune_model(
        "Kaggle Supplementary Logistic Regression",
        logistic_pipeline,
        logistic_param_grid,
        X_train,
        y_train
    )

    best_tree = tune_model(
        "Kaggle Supplementary Decision Tree",
        tree_pipeline,
        tree_param_grid,
        X_train,
        y_train
    )

    best_rf = tune_model(
        "Kaggle Supplementary Random Forest",
        rf_pipeline,
        rf_param_grid,
        X_train,
        y_train
    )

    logistic_result = evaluate_model(
        "Kaggle Supp. Logistic Regression",
        best_logistic.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    tree_result = evaluate_model(
        "Kaggle Supp. Decision Tree",
        best_tree.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    rf_result = evaluate_model(
        "Kaggle Supp. Random Forest",
        best_rf.best_estimator_,
        X_train, X_test, y_train, y_test
    )

    results = [logistic_result, tree_result, rf_result]

    save_summary(results, "kaggle_supplementary_model_summary.csv")
    plot_confusion(results, "kaggle_supplementary_confusion_matrices.png")
    plot_roc(results, y_test, "kaggle_supplementary_roc_curve.png", title="Kaggle Supplementary ROC Curve")
    save_summary(results, "kaggle_model_summary.csv")

    return {
        "results": results,
        "X": X,
        "y": y
    }


def compare_uci_and_kaggle_supplementary(uci_results, kaggle_results):
    rows = []

    for res in uci_results:
        rows.append({
            "Experiment": "UCI main experiment",
            "Model": res["name"],
            "Test Accuracy": round(res["test_acc"], 4),
            "F1-score": round(res["f1"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "ROC-AUC": round(res["auc"], 4)
        })

    for res in kaggle_results:
        rows.append({
            "Experiment": "Kaggle supplementary split",
            "Model": res["name"],
            "Test Accuracy": round(res["test_acc"], 4),
            "F1-score": round(res["f1"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "ROC-AUC": round(res["auc"], 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv("uci_vs_kaggle_supplementary_comparison.csv", index=False)
    print("\nSaved uci_vs_kaggle_supplementary_comparison.csv")
    print(df)
    return df


# -----------------------------
# true external validation
# -----------------------------
def run_true_external_validation(uci_pack):
    """
    Train models on all UCI data and evaluate directly on deduplicated,
    UCI-harmonized Kaggle data.
    """
    df = load_kaggle_data("heart_kaggle.csv", drop_dup=True)
    X_kaggle, y_kaggle = align_kaggle_columns(df, harmonize_to_uci=True)

    X_kaggle = clean_features(X_kaggle)
    y_kaggle = clean_target(y_kaggle, binary_mode="kaggle")

    common_cols = [c for c in uci_pack["X"].columns if c in X_kaggle.columns]
    print("\nCommon columns used for true external validation:", common_cols)

    X_train_full = uci_pack["X"][common_cols].copy()
    y_train_full = uci_pack["y"].copy()
    X_kaggle = X_kaggle[common_cols].copy()

    preprocessor = build_preprocessor(X_train_full)

    best_lr_params = uci_pack["best_logistic"].best_params_
    best_dt_params = uci_pack["best_tree"].best_params_
    best_rf_params = uci_pack["best_rf"].best_params_

    lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            C=best_lr_params["classifier__C"],
            random_state=RANDOM_STATE
        ))
    ])

    dt = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=best_dt_params["classifier__max_depth"],
            min_samples_split=best_dt_params["classifier__min_samples_split"],
            min_samples_leaf=best_dt_params["classifier__min_samples_leaf"],
            random_state=RANDOM_STATE
        ))
    ])

    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=best_rf_params["classifier__n_estimators"],
            max_depth=best_rf_params["classifier__max_depth"],
            min_samples_split=best_rf_params["classifier__min_samples_split"],
            min_samples_leaf=best_rf_params["classifier__min_samples_leaf"],
            random_state=RANDOM_STATE
        ))
    ])

    models = [
        ("True External LR", lr),
        ("True External DT", dt),
        ("True External RF", rf)
    ]

    rows = []
    for name, model in models:
        model.fit(X_train_full, y_train_full)

        pred = model.predict(X_kaggle)
        prob = model.predict_proba(X_kaggle)[:, 1]
        cm = confusion_matrix(y_kaggle, pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_kaggle, pred), 4),
            "F1-score": round(f1_score(y_kaggle, pred, zero_division=0), 4),
            "Precision": round(precision_score(y_kaggle, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_kaggle, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_kaggle, prob), 4),
            "False Positives": int(fp),
            "False Negatives": int(fn)
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv("true_external_validation.csv", index=False)

    print("\nSaved true_external_validation.csv")
    print(df_out)

    return df_out


# -----------------------------
# class-weighted logistic regression
# -----------------------------
def run_class_weight_experiment(X, y):
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    default_lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            C=0.01,
            random_state=RANDOM_STATE
        ))
    ])

    balanced_lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            C=0.01,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    default_lr.fit(X_train, y_train)
    balanced_lr.fit(X_train, y_train)

    results = []
    for name, model in [
        ("Default Logistic Regression", default_lr),
        ("Balanced Logistic Regression", balanced_lr)
    ]:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "F1-score": round(f1_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, prob), 4),
            "False Positives": int(fp),
            "False Negatives": int(fn)
        })

    df = pd.DataFrame(results)
    df.to_csv("class_weight_comparison.csv", index=False)
    print("\nSaved class_weight_comparison.csv")
    print(df)
    return df


# -----------------------------
# top-k feature experiment
# -----------------------------
def run_topk_feature_experiment(X, y, logistic_features, k=5):
    top_features = []
    for f in logistic_features["feature"].head(k).tolist():
        if "__" in f:
            top_features.append(f.split("__", 1)[1])
        else:
            top_features.append(f)

    top_features = [f for f in top_features if f in X.columns]

    print("\nTop-k features used:", top_features)

    X_small = X[top_features].copy()
    preprocessor = build_preprocessor(X_small)

    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    lr = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            penalty="l2",
            C=0.01,
            random_state=RANDOM_STATE
        ))
    ])

    dt = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        ))
    ])

    rf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE
        ))
    ])

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    rows = []
    for name, model in [
        ("Top-k Logistic Regression", lr),
        ("Top-k Decision Tree", dt),
        ("Top-k Random Forest", rf)
    ]:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Model": name,
            "Num Features": len(top_features),
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "F1-score": round(f1_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, prob), 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv("topk_feature_comparison.csv", index=False)
    print("\nSaved topk_feature_comparison.csv")
    print(df)
    return df


# -----------------------------
# threshold analysis
# -----------------------------
def run_threshold_experiment(best_logistic, X_test, y_test):
    prob = best_logistic.best_estimator_.predict_proba(X_test)[:, 1]
    thresholds = [0.50, 0.45, 0.40, 0.35]

    rows = []
    for t in thresholds:
        pred = (prob >= t).astype(int)
        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Threshold": t,
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "F1-score": round(f1_score(y_test, pred), 4),
            "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "False Positives": int(fp),
            "False Negatives": int(fn)
        })

    df = pd.DataFrame(rows)
    df.to_csv("threshold_analysis.csv", index=False)
    print("\nSaved threshold_analysis.csv")
    print(df)
    return df


# -----------------------------
# main
# -----------------------------
def main():
    # 1. main experiment on UCI
    uci_pack = run_main_experiment()

    # 2. supplementary Kaggle in-dataset experiment
    # This trains and tests on deduplicated Kaggle data. It is NOT true external validation.
    try:
        kaggle_pack = run_kaggle_supplementary_experiment()
        compare_uci_and_kaggle_supplementary(
            uci_pack["results"],
            kaggle_pack["results"]
        )
    except Exception as e:
        print("\nKaggle supplementary experiment skipped.")
        print("Reason:", e)

    # 3. true external validation: train on UCI, test on deduplicated and harmonized Kaggle
    try:
        run_true_external_validation(uci_pack)
    except Exception as e:
        print("\nTrue external validation skipped.")
        print("Reason:", e)

    # 4. class-weight experiment
    run_class_weight_experiment(uci_pack["X"], uci_pack["y"])

    # 5. top-k feature experiment
    run_topk_feature_experiment(
        uci_pack["X"],
        uci_pack["y"],
        uci_pack["logistic_features"],
        k=5
    )

    # 6. threshold analysis
    run_threshold_experiment(
        uci_pack["best_logistic"],
        uci_pack["X_test"],
        uci_pack["y_test"]
    )

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
