# ---------------------------------------------
# Decision Trees and Random Forest on Heart Data
# ---------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# 1. Load and Preprocess Dataset
# ---------------------------------------------
df = pd.read_csv("heart.csv")  # Replace with the downloaded file path

print("🔍 Dataset Overview:")
print(df.head())
print(df.isnull().sum())

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# 2. Decision Tree Classifier & Visualization
# ---------------------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Accuracy
y_pred_dt = dt.predict(X_test_scaled)
print("\n🌳 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Visualize Tree (textual using plot_tree)
plt.figure(figsize=(15, 6))
plot_tree(dt, feature_names=X.columns, class_names=["No", "Yes"],
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree Visualization (First 3 Levels)")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 3. Control Overfitting - Tree Depth
# ---------------------------------------------
depth_range = list(range(1, 11))
accuracy_by_depth = []

for depth in depth_range:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_by_depth.append(acc)

plt.plot(depth_range, accuracy_by_depth, marker='o')
plt.title("Accuracy vs Max Tree Depth")
plt.xlabel("Max Depth")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 4. Random Forest Classifier
# ---------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ---------------------------------------------
# 5. Feature Importances (Random Forest)
# ---------------------------------------------
importances = rf.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 6. Cross-Validation Scores
# ---------------------------------------------
dt_cv = DecisionTreeClassifier(max_depth=4, random_state=42)
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

cv_score_dt = cross_val_score(dt_cv, X, y, cv=5)
cv_score_rf = cross_val_score(rf_cv, X, y, cv=5)

print("\n📊 Cross-validation Scores:")
print(f"Decision Tree CV Accuracy (max_depth=4): {cv_score_dt.mean():.2f}")
print(f"Random Forest CV Accuracy: {cv_score_rf.mean():.2f}")
