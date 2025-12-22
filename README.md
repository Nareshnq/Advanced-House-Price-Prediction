# 🏠 Advanced House Price Prediction using Linear Regression

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict house prices using **Linear Regression and advanced data science techniques**.

The goal is not only high prediction accuracy, but also **interpretability and business insights** for real estate stakeholders.

---

## 🎯 Objectives
- Predict house prices accurately
- Handle outliers and noisy data
- Engineer meaningful features
- Compare linear and non-linear models
- Validate model assumptions
- Translate results into **business recommendations**

---

## 🗂 Dataset
- Source: USA Housing Dataset
- Size: ~5,000 records
- Target Variable: `Price`
- Features include:
  - Area Income
  - House Age
  - Number of Rooms
  - Population
  - Engineered interaction features

> ⚠️ Raw dataset not included due to licensing.
> Place dataset inside `/data` if you want to run locally.

---

## 🔍 Project Workflow

### 1️⃣ Data Exploration & EDA
- Dataset inspection and summary statistics
- Correlation analysis
- Price distribution analysis
- Visual outlier detection

### 2️⃣ Outlier Detection & Cleaning
- IQR method for detection
- Z-score method for extreme outlier removal
- Improved model stability and coefficient reliability

### 3️⃣ Feature Engineering
- Ratios (Rooms per Bedroom)
- Per-capita metrics
- Interaction terms:
  - Income × Rooms
  - Income × Age
- Categorical binning for demographic features

### 4️⃣ Feature Selection
- Recursive Feature Elimination (RFE)
- Random Forest feature importance
- Final selection based on robustness and importance

### 5️⃣ Model Training
Models evaluated:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### 6️⃣ Cross-Validation
- 5-fold cross-validation
- Ensured model stability and generalization

### 7️⃣ Hyperparameter Tuning
- GridSearchCV (streamlined grids)
- Tuned Random Forest & Gradient Boosting

### 8️⃣ Model Evaluation
Metrics used:
- MAE
- RMSE
- R² Score
- MAPE

### 9️⃣ Residual Analysis
- Residual vs predicted plot
- Q-Q plot
- Normality testing (Shapiro-Wilk)
- Validation of Linear Regression assumptions

---

## 🏆 Results

| Metric | Value |
|------|------|
| Best Model | Linear Regression |
| R² Score | ~0.91 |
| MAE | ~$82,000 |
| RMSE | ~$103,000 |
| MAPE | ~6% |

✔ Linear Regression outperformed complex models  
✔ Strong interpretability with stable performance  

---

## 📊 Key Visualizations

### Correlation & EDA
![EDA](images/eda_analysis.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

### Residual Analysis
![Residuals](images/residual_analysis.png)

---

## 💡 Business Insights
- Area income is the strongest driver of house prices
- More rooms significantly increase property value
- Income × Rooms interaction is a key price multiplier
- Linear models provide clear, explainable pricing insights

---

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SciPy

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
