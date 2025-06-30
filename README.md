# ❤️ Heart Disease Prediction with Decision Trees & Random Forests

This project demonstrates the use of **Decision Tree** and **Random Forest** classifiers on the **Heart Disease UCI Dataset**. It includes model training, visualization, feature importance analysis, overfitting control, and evaluation using cross-validation.

---

## 📊 Dataset

- **Name**: Heart Disease UCI  
- **Source**: [Kaggle Link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
- **Format**: CSV (`heart.csv`)  
- **Target**:  
  - `1` → Heart disease present  
  - `0` → No heart disease  

---

## 🧰 Tools & Libraries

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## ✅ Objectives Covered

### 1. Train a Decision Tree Classifier
- Fit a decision tree model using `DecisionTreeClassifier`
- Visualize first 3 levels of the tree using `plot_tree`

### 2. Control Overfitting with Max Depth
- Trained multiple decision trees with varying depth
- Plotted accuracy vs tree depth to find optimal complexity

### 3. Train a Random Forest Classifier
- Used `RandomForestClassifier` with 100 trees
- Compared performance with single decision tree

### 4. Feature Importance
- Extracted and visualized the most important features using `feature_importances_`

### 5. Evaluate with Cross-Validation
- Applied 5-fold cross-validation using `cross_val_score`
- Compared average accuracies of both models

---

## 📈 Performance Snapshot

| Model             | Accuracy (Test Set) | Cross-Validation Accuracy |
|------------------|---------------------|----------------------------|
| Decision Tree     | ~82–85%             | ~79% (max_depth=4)        |
| Random Forest     | ~90–93%             | ~88–91%                   |

> Note: Actual results may vary depending on train-test split.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
