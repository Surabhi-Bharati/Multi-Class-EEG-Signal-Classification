# ============================================================
# MULTI-CLASS EEG SIGNAL CLASSIFICATION USING MACHINE LEARNING
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv("BEED_Data.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()


# ============================================================
# 4. BASIC DATASET ANALYSIS
# ============================================================

print("\nClass counts:")
print(df["y"].value_counts())

print("\nClass percentages:")
print(
    df["y"].value_counts(normalize=True) * 100
)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

df["y"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(8, 5)
)

plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("BEED Class Distribution")
plt.xticks(rotation=0)

plt.savefig(
    "graphs/01_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive statistics:")
print(df.describe().T)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

correlation = df.corr()

print("\nCorrelation matrix:")
print(correlation)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of BEED Features")

plt.savefig(
    "graphs/02_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. MISSING VALUES, DUPLICATES AND UNIQUE VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique values:")
print(df.nunique())


# ============================================================
# 9. OVERALL EEG FEATURE BOXPLOTS
# ============================================================

plt.figure(figsize=(14, 6))

sns.boxplot(
    data=df.drop(columns="y")
)

plt.title("Distribution and Outliers of EEG Features")
plt.xlabel("EEG Features")
plt.ylabel("Value")
plt.xticks(rotation=45)

plt.savefig(
    "graphs/03_eeg_feature_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. FEATURES BY CLASS
# ============================================================

features = [
    f"X{i}"
    for i in range(1, 17)
]

fig, axes = plt.subplots(
    4,
    4,
    figsize=(16, 14)
)

for feature, ax in zip(
    features,
    axes.flatten()
):

    sns.boxplot(
        data=df,
        x="y",
        y=feature,
        ax=ax
    )

    ax.set_title(feature)
    ax.set_xlabel("Class")
    ax.set_ylabel("")


plt.tight_layout()

plt.savefig(
    "graphs/04_features_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. CLASS-WISE STATISTICS
# ============================================================

class_stats = df.groupby("y").agg(
    ["mean", "std"]
)

print("\nClass-wise mean and standard deviation:")
print(class_stats.T)

class_means = df.groupby("y").mean()


# ============================================================
# 12. MEAN EEG FEATURES BY CLASS
# ============================================================

plt.figure(figsize=(14, 7))

class_means.T.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title(
    "Mean EEG Feature Values by Class"
)

plt.xlabel("EEG Feature")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.legend(title="Class")

plt.tight_layout()

plt.savefig(
    "graphs/05_mean_features_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "y",
    axis=1
)

y = df["y"]


# ============================================================
# 14. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nTraining shape:")
print(X_train.shape)

print("\nValidation shape:")
print(X_val.shape)

print("\nTest shape:")
print(X_test.shape)


print("\nTraining class distribution:")
print(
    y_train.value_counts(normalize=True) * 100
)

print("\nValidation class distribution:")
print(
    y_val.value_counts(normalize=True) * 100
)

print("\nTest class distribution:")
print(
    y_test.value_counts(normalize=True) * 100
)


# ============================================================
# 15. FEATURE SCALING
# Used for Logistic Regression, KNN and SVM
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)

print("\nTraining mean:")
print(
    X_train_scaled.mean(axis=0)
)

print("\nTraining standard deviation:")
print(
    X_train_scaled.std(axis=0)
)


# ============================================================
# 16. LOGISTIC REGRESSION
# ============================================================

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_scaled,
    y_train
)

y_val_pred = lr_model.predict(
    X_val_scaled
)

lr_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

lr_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro"
)

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

print(
    "Validation Accuracy:",
    lr_accuracy
)

print(
    "Macro F1:",
    lr_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_val_pred
    )
)


cm_lr = confusion_matrix(
    y_val,
    y_val_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    "Logistic Regression - Confusion Matrix"
)

plt.savefig(
    "graphs/06_logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. K-NEAREST NEIGHBORS
# ============================================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

y_val_pred_knn = knn_model.predict(
    X_val_scaled
)

knn_accuracy = accuracy_score(
    y_val,
    y_val_pred_knn
)

knn_f1 = f1_score(
    y_val,
    y_val_pred_knn,
    average="macro"
)

print("\n========================================")
print("K-NEAREST NEIGHBORS")
print("========================================")

print(
    "Validation Accuracy:",
    knn_accuracy
)

print(
    "Macro F1:",
    knn_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_val_pred_knn
    )
)


cm_knn = confusion_matrix(
    y_val,
    y_val_pred_knn
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    "KNN - Confusion Matrix"
)

plt.savefig(
    "graphs/07_knn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Random Forest uses ORIGINAL / UNSCALED features
rf_model.fit(
    X_train,
    y_train
)

joblib.dump(
    rf_model,
    "models/rf_model.pkl"
)

print(
    "\nRandom Forest model saved successfully."
)

y_val_pred_rf = rf_model.predict(
    X_val
)

rf_accuracy = accuracy_score(
    y_val,
    y_val_pred_rf
)

rf_f1 = f1_score(
    y_val,
    y_val_pred_rf,
    average="macro"
)

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

print(
    "Validation Accuracy:",
    rf_accuracy
)

print(
    "Macro F1:",
    rf_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_val_pred_rf
    )
)


cm_rf = confusion_matrix(
    y_val,
    y_val_pred_rf
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    "Random Forest - Confusion Matrix"
)

plt.savefig(
    "graphs/08_random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Importance")
plt.ylabel("EEG Feature")

plt.tight_layout()

plt.savefig(
    "graphs/10_random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. SVM
# ============================================================

svm_model = SVC(
    kernel="rbf",
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)

y_val_pred_svm = svm_model.predict(
    X_val_scaled
)

svm_accuracy = accuracy_score(
    y_val,
    y_val_pred_svm
)

svm_f1 = f1_score(
    y_val,
    y_val_pred_svm,
    average="macro"
)

print("\n========================================")
print("SVM")
print("========================================")

print(
    "Validation Accuracy:",
    svm_accuracy
)

print(
    "Macro F1:",
    svm_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_val_pred_svm
    )
)


cm_svm = confusion_matrix(
    y_val,
    y_val_pred_svm
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    "SVM - Confusion Matrix"
)

plt.savefig(
    "graphs/09_svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. XGBOOST
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)

# XGBoost uses ORIGINAL / UNSCALED features
xgb_model.fit(
    X_train,
    y_train
)
# Saving XGBoost model
joblib.dump(
    xgb_model,
    "models/xgb_model.pkl"
)

print("XGBoost model saved successfully.")
y_val_pred_xgb = xgb_model.predict(
    X_val
)

xgb_accuracy = accuracy_score(
    y_val,
    y_val_pred_xgb
)

xgb_f1 = f1_score(
    y_val,
    y_val_pred_xgb,
    average="macro"
)

print("\n========================================")
print("XGBOOST")
print("========================================")

print(
    "Validation Accuracy:",
    xgb_accuracy
)

print(
    "Macro F1:",
    xgb_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_val_pred_xgb
    )
)


cm_xgb = confusion_matrix(
    y_val,
    y_val_pred_xgb
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(
    "XGBoost - Confusion Matrix"
)

plt.savefig(
    "graphs/14_xgboost_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. FIVE-MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM",
        "XGBoost"
    ],

    "Accuracy": [
        lr_accuracy,
        knn_accuracy,
        rf_accuracy,
        svm_accuracy,
        xgb_accuracy
    ],

    "Macro F1": [
        lr_f1,
        knn_f1,
        rf_f1,
        svm_f1,
        xgb_f1
    ]
})


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(results)


# ============================================================
# 23. SAVE MODEL COMPARISON
# ============================================================

results.to_csv(
    "model_comparison_results.csv",
    index=False
)


# ============================================================
# 24. GAUSSIAN NOISE FUNCTION
# ============================================================

def add_gaussian_noise(
    X,
    noise_level,
    random_state=42
):

    rng = np.random.default_rng(
        random_state
    )

    noise = rng.normal(
        loc=0,
        scale=noise_level * X.std(axis=0),
        size=X.shape
    )

    return X + noise


# ============================================================
# 25. GAUSSIAN NOISE ANALYSIS
# ============================================================

noise_levels = [
    0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]

noise_results = []


for noise in noise_levels:

    X_val_noisy = add_gaussian_noise(
        X_val,
        noise
    )

    # Scale only for scaled models
    X_val_noisy_scaled = scaler.transform(
        X_val_noisy
    )

    # Logistic Regression
    pred_lr = lr_model.predict(
        X_val_noisy_scaled
    )

    # KNN
    pred_knn = knn_model.predict(
        X_val_noisy_scaled
    )

    # Random Forest
    pred_rf = rf_model.predict(
        X_val_noisy
    )

    # SVM
    pred_svm = svm_model.predict(
        X_val_noisy_scaled
    )

    # XGBoost
    pred_xgb = xgb_model.predict(
        X_val_noisy
    )

    noise_results.append({

        "Noise Level": noise * 100,

        "Logistic Regression":
            accuracy_score(
                y_val,
                pred_lr
            ),

        "KNN":
            accuracy_score(
                y_val,
                pred_knn
            ),

        "Random Forest":
            accuracy_score(
                y_val,
                pred_rf
            ),

        "SVM":
            accuracy_score(
                y_val,
                pred_svm
            ),

        "XGBoost":
            accuracy_score(
                y_val,
                pred_xgb
            )
    })


noise_results = pd.DataFrame(
    noise_results
)

print("\n========================================")
print("GAUSSIAN NOISE RESULTS")
print("========================================")

print(noise_results)


# Save Gaussian noise results
noise_results.to_csv(
    "gaussian_noise_results.csv",
    index=False
)


# ============================================================
# 26. GAUSSIAN NOISE PERFORMANCE GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    noise_results["Noise Level"],
    noise_results["Logistic Regression"],
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    noise_results["Noise Level"],
    noise_results["KNN"],
    marker="o",
    label="KNN"
)

plt.plot(
    noise_results["Noise Level"],
    noise_results["Random Forest"],
    marker="o",
    label="Random Forest"
)

plt.plot(
    noise_results["Noise Level"],
    noise_results["SVM"],
    marker="o",
    label="SVM"
)

plt.plot(
    noise_results["Noise Level"],
    noise_results["XGBoost"],
    marker="o",
    label="XGBoost"
)

plt.xlabel(
    "Gaussian Noise Level (%)"
)

plt.ylabel(
    "Validation Accuracy"
)

plt.title(
    "Model Performance Under Increasing EEG Gaussian Noise"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "graphs/11_gaussian_noise_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 27. IMPULSE NOISE FUNCTION
# ============================================================

def add_impulse_noise(
    X,
    noise_level,
    random_state=42
):

    X_noisy = X.astype(float).copy()

    rng = np.random.default_rng(
        random_state
    )

    n_samples, n_features = X_noisy.shape

    n_values = int(
        noise_level
        * n_samples
        * n_features
    )

    rows = rng.integers(
        0,
        n_samples,
        n_values
    )

    cols = rng.integers(
        0,
        n_features,
        n_values
    )

    feature_std = X_noisy.std()

    for row, col in zip(
        rows,
        cols
    ):

        spike = (
            rng.choice([-1, 1])
            * 5
            * feature_std.iloc[col]
        )

        X_noisy.iloc[
            row,
            col
        ] += spike

    return X_noisy


# ============================================================
# 28. IMPULSE NOISE ANALYSIS
# ============================================================

impulse_results = []


for noise in noise_levels:

    X_val_noisy = add_impulse_noise(
        X_val,
        noise
    )

    # Scale only for scaled models
    X_val_noisy_scaled = scaler.transform(
        X_val_noisy
    )

    # Logistic Regression
    lr_pred = lr_model.predict(
        X_val_noisy_scaled
    )

    # KNN
    knn_pred = knn_model.predict(
        X_val_noisy_scaled
    )

    # IMPORTANT:
    # Random Forest was trained on unscaled features
    rf_pred = rf_model.predict(
        X_val_noisy
    )

    # SVM
    svm_pred = svm_model.predict(
        X_val_noisy_scaled
    )

    # XGBoost was trained on unscaled features
    xgb_pred = xgb_model.predict(
        X_val_noisy
    )

    impulse_results.append({

        "Noise Level": noise * 100,

        "Logistic Regression":
            accuracy_score(
                y_val,
                lr_pred
            ),

        "KNN":
            accuracy_score(
                y_val,
                knn_pred
            ),

        "Random Forest":
            accuracy_score(
                y_val,
                rf_pred
            ),

        "SVM":
            accuracy_score(
                y_val,
                svm_pred
            ),

        "XGBoost":
            accuracy_score(
                y_val,
                xgb_pred
            )
    })


impulse_results = pd.DataFrame(
    impulse_results
)

print("\n========================================")
print("IMPULSE NOISE RESULTS")
print("========================================")

print(impulse_results)


# Save impulse noise results
impulse_results.to_csv(
    "impulse_noise_results.csv",
    index=False
)


# ============================================================
# 29. IMPULSE NOISE PERFORMANCE GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    impulse_results["Noise Level"],
    impulse_results["Logistic Regression"],
    marker="o",
    label="Logistic Regression"
)

plt.plot(
    impulse_results["Noise Level"],
    impulse_results["KNN"],
    marker="o",
    label="KNN"
)

plt.plot(
    impulse_results["Noise Level"],
    impulse_results["Random Forest"],
    marker="o",
    label="Random Forest"
)

plt.plot(
    impulse_results["Noise Level"],
    impulse_results["SVM"],
    marker="o",
    label="SVM"
)

plt.plot(
    impulse_results["Noise Level"],
    impulse_results["XGBoost"],
    marker="o",
    label="XGBoost"
)

plt.xlabel(
    "Impulse Noise Level (%)"
)

plt.ylabel(
    "Validation Accuracy"
)

plt.title(
    "Model Performance Under Increasing EEG Impulse Noise"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "graphs/12_impulse_noise_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 30. FINAL RANDOM FOREST TEST EVALUATION
# ============================================================

rf_test_pred = rf_model.predict(
    X_test
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_pred
)

rf_test_f1 = f1_score(
    y_test,
    rf_test_pred,
    average="macro"
)


print("\n========================================")
print("FINAL RANDOM FOREST TEST EVALUATION")
print("========================================")

print(
    "Random Forest Test Accuracy:",
    rf_test_accuracy
)

print(
    "Random Forest Test Macro F1:",
    rf_test_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_test_pred
    )
)


# ============================================================
# 31. RANDOM FOREST TEST CONFUSION MATRIX
# ============================================================

cm_rf_test = confusion_matrix(
    y_test,
    rf_test_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_rf_test,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    "Random Forest - Final Test Confusion Matrix"
)

plt.savefig(
    "graphs/13_random_forest_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 32. FINAL XGBOOST TEST EVALUATION
# ============================================================

xgb_test_pred = xgb_model.predict(
    X_test
)

xgb_test_accuracy = accuracy_score(
    y_test,
    xgb_test_pred
)

xgb_test_f1 = f1_score(
    y_test,
    xgb_test_pred,
    average="macro"
)


print("\n========================================")
print("FINAL XGBOOST TEST EVALUATION")
print("========================================")

print(
    "XGBoost Test Accuracy:",
    xgb_test_accuracy
)

print(
    "XGBoost Test Macro F1:",
    xgb_test_f1
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_test_pred
    )
)


# ============================================================
# 33. XGBOOST TEST CONFUSION MATRIX
# ============================================================

cm_xgb_test = confusion_matrix(
    y_test,
    xgb_test_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_xgb_test,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title(
    "XGBoost - Final Test Confusion Matrix"
)

plt.savefig(
    "graphs/15_xgboost_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 34. FINAL MODEL SUMMARY
# ============================================================

print("\n========================================")
print("FINAL MODEL SUMMARY")
print("========================================")

final_comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM",
        "XGBoost"
    ],

    "Validation Accuracy": [
        lr_accuracy,
        knn_accuracy,
        rf_accuracy,
        svm_accuracy,
        xgb_accuracy
    ],

    "Validation Macro F1": [
        lr_f1,
        knn_f1,
        rf_f1,
        svm_f1,
        xgb_f1
    ]
})


print(final_comparison)


print("\n----------------------------------------")
print("FINAL TEST RESULTS")
print("----------------------------------------")

print(
    "Random Forest Test Accuracy:",
    rf_test_accuracy
)

print(
    "Random Forest Test Macro F1:",
    rf_test_f1
)

print(
    "XGBoost Test Accuracy:",
    xgb_test_accuracy
)

print(
    "XGBoost Test Macro F1:",
    xgb_test_f1
)


# ============================================================
# 35. DETERMINING BEST VALIDATION MODEL
# ============================================================

best_model_row = final_comparison.loc[
    final_comparison["Validation Accuracy"].idxmax()
]

print("\n----------------------------------------")
print("BEST VALIDATION MODEL")
print("----------------------------------------")

print(
    "Model:",
    best_model_row["Model"]
)

print(
    "Validation Accuracy:",
    best_model_row["Validation Accuracy"]
)

print(
    "Validation Macro F1:",
    best_model_row["Validation Macro F1"]
)


# ============================================================
# 36. COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("ALL ANALYSIS COMPLETED SUCCESSFULLY")
print("========================================")

print("\nGenerated folders/files:")

print("graphs/")
print("models/rf_model.pkl")
print("model_comparison_results.csv")
print("gaussian_noise_results.csv")
print("impulse_noise_results.csv")