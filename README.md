# Crop Recommendation under Cost Constraints

This repository implements a production-ready machine learning pipeline for agricultural crop recommendation. The system is designed with a strict real-world business constraint: optimizing for cost by minimizing the number of expensive soil tests required.

## Problem Statement
Farmers need a reliable crop recommendation system based on soil and environmental metrics. The dataset includes:
- **Free weather metrics**: Temperature, Humidity, Rainfall (assumed fetchable via free weather APIs).
- **Expensive soil metrics**: Nitrogen (N), Phosphorus (P), Potassium (K), and pH.

**Constraint:** The farmer can only afford the equipment/lab test to measure *one* of the four soil attributes.

## Solution Architecture
To solve this constraint, our system features an automated **Feature Selection** step to securely identify the single most predictive soil feature before training the final model.

### 1. Data Processing
- Handles missing values using median imputation.
- Isolates inputs to prevent data leakage before standard scaling.
- Applies Label Encoding for the string target output.

### 2. Feature Selection (Finding the Best Soil Metric)
The pipeline compares two robust feature selection methods to pick the absolute best soil feature among `N, P, K, pH`:
1. **Statistical (ANOVA F-Statistic):** Identifies features whose means vary significantly across different crops (strong linear discriminator).
2. **Algorithmic (Random Forest Importance):** Calculates Gini impurity reduction across hundreds of decision trees to capture complex, non-linear relationships.

By default, the pipeline uses the Random Forest's top feature combined with the 3 free weather metrics.

### 3. Model Training & Tuning
- **Algorithm:** Multinomial Logistic Regression (`solver='saga'`).
- **Hyperparameter Tuning:** Uses `GridSearchCV` wrapped in `StratifiedKFold` (5 folds) to find the optimal L2 regularization strength (`C`).
- Extracts test features, applies `StandardScaler`, and evaluates accuracy.

## Repository Contents
- `crop_recommendation.py` - The complete, object-oriented pipeline script.
- `README.md` - Project documentation.
- `output/` - Auto-generated directory (created when running the script) that stores:
  - `feature_selection_comparison.png` (Bar charts of feature importance)
  - `confusion_matrix.png` (Heatmap of model performance)
  - `crop_recommendation_model.joblib` (Serialized model)
  - `scaler.joblib`, `label_encoder.joblib` (Serialized preprocessors)

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/devhemanthac-commits/Crop-Recommendation-Model.git
   cd Crop-Recommendation-Model
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

3. **Add the dataset:**
   - Download the "Crop Recommendation Dataset" from Kaggle.
   - Place `Crop_recommendation.csv` in the root directory.

4. **Execute the pipeline:**
   ```bash
   python crop_recommendation.py
   ```

## Output Logs
Running the script provides detailed console logging out-of-the-box, including CV scores, a full classification report (precision, recall, f1-score per crop), and final test accuracy. All visualizations and models will automatically be serialized to the `output/` folder for production deployment.