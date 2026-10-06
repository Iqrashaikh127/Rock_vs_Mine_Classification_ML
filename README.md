# Rock vs Mine Classification using Machine Learning

## Project Overview

This project uses machine learning to classify sonar signals as either **Rock** or **Mine**.

The model analyzes sonar frequency measurements and learns patterns that help distinguish between the two classes.

The project covers data preprocessing, exploratory data analysis, dimensionality reduction, model comparison, model tuning, evaluation, and deployment using Streamlit.

## Objective

To build a machine learning classification model that can accurately predict whether a sonar signal represents a **Rock** or a **Mine**.

- `0` → Rock
- `1` → Mine

## Dataset

The dataset contains **3,000 records** with **60 sonar signal features** and one target column.

Each feature represents a sonar frequency measurement used for classification.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Workflow

### 1. Data Preprocessing

The dataset was analyzed and prepared for machine learning by:

- Handling missing values using median imputation
- Converting the target into numerical values
- Applying Yeo-Johnson transformation to reduce skewness
- Standardizing the features using StandardScaler

### 2. Dimensionality Reduction

The dataset originally contained **60 features**.

PCA (Principal Component Analysis) was applied while retaining approximately **95% of the variance**, reducing the input features from **60 to 19 components**.

### 3. Model Comparison

The following machine learning algorithms were compared:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

SVM achieved the best cross-validation performance and was selected for further tuning.

### 4. Hyperparameter Tuning

GridSearchCV was used to find the best SVM parameters.

**Best Parameters:**

- C = 10
- Kernel = RBF
- Gamma = Scale

The tuned SVM achieved a cross-validation F1 score of **93.87%**.

## Model Performance

The final SVM model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 94.67% |
| Precision | 94.56% |
| Recall | 95.72% |
| F1 Score | 95.14% |
| ROC-AUC | 99.03% |

These results indicate that the model performs well in distinguishing between Rock and Mine signals.

## Streamlit Deployment

The trained model was deployed using **Streamlit**, allowing users to provide sonar signal inputs and receive a prediction.

The application returns the predicted class:

- **Rock**
- **Mine**

## 👩‍💻 Author

**Iqra Shaikh**

Data Science | Data Analytics | AI/ML
