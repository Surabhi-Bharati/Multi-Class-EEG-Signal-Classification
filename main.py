import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("graphs", exist_ok=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("BEED_Data.csv")
print(df.head())
print(df.shape)
df.info()
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True) * 100)
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
print(df.describe().T)
correlation = df.corr()
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
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.nunique())
plt.figure(figsize=(14, 6))

sns.boxplot(data=df.drop(columns="y"))

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
features = [f"X{i}" for i in range(1, 17)]

fig, axes = plt.subplots(4, 4, figsize=(16, 14))

for feature, ax in zip(features, axes.flatten()):
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
class_stats = df.groupby("y").agg(["mean", "std"])
print(class_stats.T)
class_means = df.groupby("y").mean()

plt.figure(figsize=(14, 7))

class_means.T.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Mean EEG Feature Values by Class")
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
X = df.drop("y", axis=1)
y = df["y"]
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
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation class distribution:")
print(y_val.value_counts(normalize=True) * 100)

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True) * 100)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training mean:")
print(X_train_scaled.mean(axis=0))

print("\nTraining standard deviation:")
print(X_train_scaled.std(axis=0))

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

y_val_pred = lr_model.predict(X_val_scaled)

accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.savefig(
    "graphs/06_logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    rf_model,
    "models/rf_model.pkl"
)

print("Random Forest model saved successfully.")
y_val_pred_rf = rf_model.predict(X_val)

print("Random Forest Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_rf))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_rf))
cm_rf = confusion_matrix(y_val, y_val_pred_rf)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Random Forest - Confusion Matrix")

plt.savefig(
    "graphs/08_random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)
plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("EEG Feature")

plt.tight_layout()
plt.savefig(
    "graphs/10_random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

y_val_pred_knn = knn_model.predict(X_val_scaled)

print("KNN Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_knn))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_knn))
cm_knn = confusion_matrix(y_val, y_val_pred_knn)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("KNN - Confusion Matrix")

plt.savefig(
    "graphs/07_knn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
from sklearn.metrics import accuracy_score, f1_score

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_val, y_val_pred),
        accuracy_score(y_val, y_val_pred_knn),
        accuracy_score(y_val, y_val_pred_rf)
    ],
    "Macro F1": [
        f1_score(y_val, y_val_pred, average="macro"),
        f1_score(y_val, y_val_pred_knn, average="macro"),
        f1_score(y_val, y_val_pred_rf, average="macro")
    ]
})

print(results)
from sklearn.svm import SVC

svm_model = SVC(
    kernel="rbf",
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

y_val_pred_svm = svm_model.predict(X_val_scaled)

print("SVM Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_svm))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_svm))
cm_svm = confusion_matrix(y_val, y_val_pred_svm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("SVM - Confusion Matrix")

plt.savefig(
    "graphs/09_svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

clean_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        accuracy_score(y_val, y_val_pred),
        accuracy_score(y_val, y_val_pred_knn),
        accuracy_score(y_val, y_val_pred_rf),
        accuracy_score(y_val, y_val_pred_svm)
    ],
    "Macro F1": [
        f1_score(y_val, y_val_pred, average="macro"),
        f1_score(y_val, y_val_pred_knn, average="macro"),
        f1_score(y_val, y_val_pred_rf, average="macro"),
        f1_score(y_val, y_val_pred_svm, average="macro")
    ]
})

print(clean_results)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def add_gaussian_noise(X, noise_level, random_state=42):
    rng = np.random.default_rng(random_state)

    noise = rng.normal(
        loc=0,
        scale=noise_level * X.std(axis=0),
        size=X.shape
    )

    return X + noise
noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

noise_results = []

for noise in noise_levels:

    X_val_noisy = add_gaussian_noise(
        X_val,
        noise
    )

    # Scale noisy validation data
    X_val_noisy_scaled = scaler.transform(X_val_noisy)

    # Predictions
    pred_lr = lr_model.predict(X_val_noisy_scaled)
    pred_knn = knn_model.predict(X_val_noisy_scaled)
    pred_rf = rf_model.predict(X_val_noisy)
    pred_svm = svm_model.predict(X_val_noisy_scaled)

    noise_results.append({
        "Noise Level": noise * 100,
        "Logistic Regression": accuracy_score(y_val, pred_lr),
        "KNN": accuracy_score(y_val, pred_knn),
        "Random Forest": accuracy_score(y_val, pred_rf),
        "SVM": accuracy_score(y_val, pred_svm)
    })

noise_results = pd.DataFrame(noise_results)

print(noise_results)
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

plt.xlabel("Gaussian Noise Level (%)")
plt.ylabel("Validation Accuracy")

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

def add_impulse_noise(X, noise_level, random_state=42):

    # Convert to float so we can safely add decimal noise
    X_noisy = X.astype(float).copy()

    rng = np.random.default_rng(random_state)

    n_samples, n_features = X_noisy.shape

    # Number of values to corrupt
    n_values = int(noise_level * n_samples * n_features)

    # Random positions
    rows = rng.integers(0, n_samples, n_values)
    cols = rng.integers(0, n_features, n_values)

    # Standard deviation of each feature
    feature_std = X_noisy.std()

    # Add large positive/negative spikes
    for row, col in zip(rows, cols):

        spike = rng.choice([-1, 1]) * 5 * feature_std.iloc[col]

        X_noisy.iloc[row, col] += spike

    return X_noisy
noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

impulse_results = []

for noise in noise_levels:

    X_val_noisy = add_impulse_noise(
        X_val,
        noise
    )

    # Scale noisy validation data
    X_val_noisy_scaled = scaler.transform(X_val_noisy)

    # Predictions
    lr_pred = lr_model.predict(X_val_noisy_scaled)
    knn_pred = knn_model.predict(X_val_noisy_scaled)
    rf_pred = rf_model.predict(X_val_noisy_scaled)
    svm_pred = svm_model.predict(X_val_noisy_scaled)

    # Store accuracies
    impulse_results.append({
        "Noise Level": noise * 100,
        "Logistic Regression": accuracy_score(y_val, lr_pred),
        "KNN": accuracy_score(y_val, knn_pred),
        "Random Forest": accuracy_score(y_val, rf_pred),
        "SVM": accuracy_score(y_val, svm_pred)
    })

impulse_results = pd.DataFrame(impulse_results)

print("\nImpulse Noise Results:")
print(impulse_results)
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

plt.xlabel("Impulse Noise Level (%)")
plt.ylabel("Validation Accuracy")

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
rf_test_pred = rf_model.predict(X_test)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_pred
)

print("Random Forest Test Accuracy:", rf_test_accuracy)
print(
    classification_report(
        y_test,
        rf_test_pred
    )
)
cm = confusion_matrix(
    y_test,
    rf_test_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Random Forest - Final Test Confusion Matrix")

plt.savefig(
    "graphs/13_random_forest_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()