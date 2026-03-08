import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings('ignore')

class CropRecommendationSystem:
    """
    A robust, production-ready system for recommending crops based on 
    budget-constrained soil metrics and free weather data.
    """
    def __init__(self, data_path: str, output_dir: str = 'output'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.winning_soil_feature = None
        self.weather_features = ['temperature', 'humidity', 'rainfall']
        self.soil_features = ['N', 'P', 'K', 'ph']
        self.final_features = []
        
        # Create output directory for artifacts (plots, models)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Loads data, handles constraints, and encodes labels."""
        logging.info("Starting Data Loading & Preprocessing...")
        try:
            df = pd.read_csv(self.data_path)
            logging.info(f"Dataset successfully loaded with shape: {df.shape}")
        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.data_path}")
            raise

        # Sanity check for required columns
        required_cols = self.weather_features + self.soil_features + ['label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # Handle missing values robustly
        if df.isnull().sum().any():
            logging.warning("Missing values detected. Imputing numerical features with median.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            df.dropna(subset=['label'], inplace=True)

        X = df.drop(columns=['label'])
        y = self.label_encoder.fit_transform(df['label'])
        
        return X, y

    def select_best_soil_feature(self, X: pd.DataFrame, y: np.ndarray) -> str:
        """
        Determines the single most predictive soil feature using Statistical (ANOVA) 
        and Algorithmic (Random Forest) approaches.
        """
        logging.info("Starting Feature Selection (Budget Constraint Simulation)...")
        X_soil = X[self.soil_features]

        # 1. ANOVA F-statistic
        f_stats, p_values = f_classif(X_soil, y)
        anova_results = pd.Series(f_stats, index=self.soil_features).sort_values(ascending=False)
        logging.info(f"ANOVA F-statistics:\n{anova_results.to_string()}")

        # 2. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_soil, y)
        rf_results = pd.Series(rf.feature_importances_, index=self.soil_features).sort_values(ascending=False)
        logging.info(f"Random Forest Importances:\n{rf_results.to_string()}")

        # Visualize feature importances
        self._plot_feature_selection(anova_results, rf_results)

        # Decision Logic: We favor the Random Forest interpretation as tree-based ensembles 
        # naturally handle non-linear real-world interactions better than linear ANOVA.
        self.winning_soil_feature = rf_results.index[0]
        logging.info(f"Winning Soil Feature Selected: '{self.winning_soil_feature}'")
        
        return self.winning_soil_feature

    def _plot_feature_selection(self, anova_results: pd.Series, rf_results: pd.Series):
        """Generates and saves a comparative plot for feature selection results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.barplot(x=anova_results.values, y=anova_results.index, ax=axes[0], palette="viridis")
        axes[0].set_title('ANOVA F-Statistic')
        axes[0].set_xlabel('F-Score')
        
        sns.barplot(x=rf_results.values, y=rf_results.index, ax=axes[1], palette="magma")
        axes[1].set_title('Random Forest Importance')
        axes[1].set_xlabel('Gini Importance')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'feature_selection_comparison.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Feature selection plot saved to {plot_path}")

    def train_and_tune_model(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Isolates allowable features, scales them, and uses GridSearchCV to find 
        the optimal hyperparameters for Logistic Regression.
        """
        logging.info("Preparing final feature set and isolating constraints...")
        self.final_features = self.weather_features + [self.winning_soil_feature]
        X_final = X[self.final_features]

        # Stratified split ensures rare crops hold minority representation in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logging.info("Tuning Logistic Regression via GridSearchCV...")
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'] # 'saga' supports l2, l1, elasticnet, but l2 is a robust default
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        base_model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=2000, random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        logging.info(f"Best hyperparameters found: {grid_search.best_params_}")
        logging.info(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate_model(self, X_test_scaled: np.ndarray, y_test: np.ndarray):
        """Evaluates the model on the hold-out test set and saves a confusion matrix."""
        logging.info("Evaluating optimal model on unseen test data...")
        y_pred = self.model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"==> Final Test Accuracy: {acc:.4f} ({acc * 100:.2f}%) <==")

        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("\n" + "="*50)
        print("Detailed Classification Report")
        print("="*50)
        print(report)

        # Generate Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Crop Recommendation')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300)
        plt.close()
        logging.info(f"Confusion matrix plot saved to {cm_path}")

    def save_artifacts(self):
        """Saves the trained model, scaler, and label encoder for deployment."""
        if self.model is None:
            logging.error("Model has not been trained yet. Cannot save.")
            return

        model_path = os.path.join(self.output_dir, 'crop_recommendation_model.joblib')
        scaler_path = os.path.join(self.output_dir, 'scaler.joblib')
        encoder_path = os.path.join(self.output_dir, 'label_encoder.joblib')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        logging.info("Model, Scaler, and LabelEncoder successfully serialized to disk.")

def main():
    # Filepath configuration
    DATA_FILE = "Crop_recommendation.csv"
    
    print("\n" + "*"*60)
    print(" ADVANCED CROP RECOMMENDATION PIPELINE ".center(60, '*'))
    print("*"*60 + "\n")

    try:
        # Initialize the system
        system = CropRecommendationSystem(data_path=DATA_FILE)
        
        # 1. Load & Preprocess
        X, y = system.load_and_preprocess()
        
        # 2. Strict Feature Selection (Budget constraint)
        # Assuming we need to run Feature Selection on the whole dataset to find 
        # the overall absolute best feature before splitting for final model training.
        # Strict evaluation could also do this on X_train only, but system design determines it here.
        system.select_best_soil_feature(X, y)
        
        # 3. Train & Tune (Cross-validated Grid Search)
        _, X_test_scaled, _, y_test = system.train_and_tune_model(X, y)
        
        # 4. Rigorous Evaluation
        system.evaluate_model(X_test_scaled, y_test)
        
        # 5. Serialization for Production
        system.save_artifacts()
        
        logging.info("Pipeline execution completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
