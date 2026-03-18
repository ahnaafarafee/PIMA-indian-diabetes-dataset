# Diabetes Prediction Project Overview

## Objective
The goal of this project is to build a machine learning model that can predict whether a patient is likely to have diabetes based on diagnostic health measurements.

---

## Dataset
The dataset used in this project was collected from **Kaggle**. It is the **Pima Indians Diabetes Dataset**, which is commonly used for binary classification problems in healthcare.

The dataset contains several medical predictor variables, such as:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is:

- **Outcome**  
  - `0` = No diabetes  
  - `1` = Diabetes

---

## What I Did in This Project

### 1. Data Loading and Initial Exploration
I first loaded the dataset using Pandas and performed an initial inspection to understand:

- the number of rows and columns
- data types
- summary statistics
- missing values
- duplicate rows
- target class distribution

This helped me understand the structure and quality of the dataset before modeling.

---

### 2. Exploratory Data Analysis (EDA)
To better understand the dataset, I performed exploratory analysis including:

- checking summary statistics
- identifying duplicate rows
- checking missing values
- visualizing class imbalance in the target variable
- plotting boxplots to inspect outliers
- comparing feature distributions against the target class
- generating a correlation heatmap

This step helped identify patterns in the data and detect potential preprocessing needs before training the models.

---

## Data Cleaning and Preprocessing

### Handling Invalid Zero Values
Some columns contained zero values that are not realistic in a medical context and may represent missing data. I treated zero values as missing in the following columns:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- Age

These zero values were replaced with `NaN` so they could be handled properly during preprocessing.

### Missing Value Imputation
Instead of dropping rows, I used **median imputation** to fill missing values. Median imputation is a good choice because it is more robust to outliers than mean imputation.

### Train-Test Split
I separated the data into:

- **Features (`X`)**
- **Target (`y`)**

Then I split the dataset into training and testing sets using an **80/20 split** with **stratification** to preserve the class distribution.

---

## Feature Engineering
Feature engineering in this project was mainly focused on **data preparation and transformation** rather than creating entirely new variables.

The feature engineering steps included:

### 1. Converting Invalid Zeros into Missing Values
This improved data quality by treating suspicious zero values as missing instead of valid measurements.

### 2. Median Imputation
Missing values were filled using the median of each feature.

### 3. Feature Scaling
For Logistic Regression, I applied **StandardScaler** so that all features would be on a similar scale. This is important because Logistic Regression is sensitive to feature magnitude.

### Note
I did **not create new handcrafted features** such as BMI categories, glucose-to-age ratios, or interaction terms. So the feature engineering in this notebook is primarily preprocessing-based.

---

## Models Used

## 1. Logistic Regression
I first built a **baseline Logistic Regression model** using a pipeline with:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `LogisticRegression(max_iter=1000, class_weight="balanced")`

### Why Logistic Regression?
Logistic Regression is a strong baseline model for binary classification because it is:

- simple
- interpretable
- efficient
- commonly used in medical prediction tasks

I used `class_weight="balanced"` to help address class imbalance in the dataset.

---

## 2. XGBoost Classifier
After building the baseline model, I trained an **XGBoost classifier** using another pipeline with:

- `SimpleImputer(strategy="median")`
- `XGBClassifier(...)`

The XGBoost model was initialized with parameters such as:

- `n_estimators=500`
- `learning_rate=0.05`
- `max_depth=3`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `reg_lambda=1.0`

### Why XGBoost?
XGBoost is a powerful gradient boosting algorithm that often performs very well on structured/tabular datasets. It can capture more complex relationships than Logistic Regression.

---

## Model Tuning

## Logistic Regression Tuning
I tuned the Logistic Regression model using **GridSearchCV**.

### Hyperparameters tuned:
- `penalty`: `l1`, `l2`
- `C`: `0.01, 0.1, 1, 10, 100`
- `solver`: `liblinear`
- `class_weight`: `None`, `balanced`

### Tuning Strategy
- **Cross-validation:** 5-fold
- **Scoring metric:** `roc_auc`

This helped identify the best combination of parameters for Logistic Regression based on ROC-AUC.

---

## XGBoost Tuning
I tuned the XGBoost model using **RandomizedSearchCV**.

### Hyperparameters tuned:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `reg_lambda`

### Tuning Strategy
- **Number of random combinations tested:** 30
- **Cross-validation:** 5-fold
- **Scoring metric:** `roc_auc`

RandomizedSearchCV was used here because XGBoost has a larger search space, and randomized search is more efficient than testing every possible combination.

---

## Evaluation Metrics
To evaluate model performance, I used:

- **Confusion Matrix**
- **Classification Report**
  - Precision
  - Recall
  - F1-score
- **ROC-AUC Score**

### Why ROC-AUC?
ROC-AUC is especially useful for binary classification because it measures how well the model distinguishes between the two classes across different thresholds.

---

## Summary of the Workflow
In summary, this project followed these steps:

1. Loaded the diabetes dataset from Kaggle
2. Performed exploratory data analysis
3. Cleaned the dataset and handled invalid zero values
4. Imputed missing values using the median
5. Scaled features for Logistic Regression
6. Built a baseline Logistic Regression model
7. Tuned Logistic Regression with GridSearchCV
8. Built an XGBoost model
9. Tuned XGBoost with RandomizedSearchCV
10. Evaluated and compared model performance using ROC-AUC and classification metrics

---

## Final Notes
This project shows a complete end-to-end machine learning workflow for a healthcare classification problem. It includes:

- dataset collection from Kaggle
- data cleaning
- preprocessing
- feature engineering through transformation
- model training
- hyperparameter tuning
- final evaluation

Among the tested models, **XGBoost performed best overall**, making it the final selected model for this diabetes prediction task.