import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
    
    def train_decision_tree(self, X_train, y_train, **kwargs):
        """Train Decision Tree classifier"""
        print("Training Decision Tree...")
        dt_params = {
            'random_state': self.random_state,
            **kwargs
        }
        dt_model = DecisionTreeClassifier(**dt_params)
        dt_model.fit(X_train, y_train)
        self.models['decision_tree'] = dt_model
        return dt_model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest classifier"""
        print("Training Random Forest...")
        rf_params = {
            'n_estimators': 200,
            'random_state': self.random_state,
            'class_weight': 'balanced',
            **kwargs
        }
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        return rf_model
    
    def train_catboost(self, X_train, y_train, **kwargs):
        """Train CatBoost classifier"""
        print("Training CatBoost...")
        cb_params = {
            'iterations': 300,
            'depth': 8,
            'learning_rate': 0.05,
            'random_seed': self.random_state,
            'verbose': 0,
            **kwargs
        }
        cat_model = CatBoostClassifier(**cb_params)
        cat_model.fit(X_train, y_train)
        self.models['catboost'] = cat_model
        return cat_model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost classifier"""
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.05,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss',
            **kwargs
        }
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        return xgb_model
    
    def predict_all(self, X_test):
        """Generate predictions for all trained models"""
        for name, model in self.models.items():
            self.predictions[name] = model.predict(X_test)
        return self.predictions
    
    def evaluate_model(self, model_name, y_true, y_pred):
        """Evaluate a single model"""
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(report)
        
        return accuracy, report
    
    def save_model(self, model_name, file_path):
        """Save trained model to file"""
        if model_name in self.models:
            joblib.dump(self.models[model_name], file_path)
            print(f"Model saved to {file_path}")
        else:
            print(f"Model {model_name} not found!")
    
    def save_all_models(self, base_path=""):
        """Save all trained models"""
        for name in self.models:
            file_path = f"{base_path}{name}_model.pkl"
            self.save_model(name, file_path)

def create_validation_samples():
    """
    Create validation samples for model testing
    """
    validation_samples = np.array([
        # Good (Low Risk)
        [2, 25, 115, 72, 160, 21.5, 0, 0],
        [1, 30, 120, 78, 170, 22.5, 0, 0],
        [2, 35, 118, 75, 165, 23.0, 0, 0],
        
        # Fair (Medium Risk)
        [2, 40, 130, 85, 200, 26.0, 0, 0],
        [1, 37, 128, 83, 195, 25.0, 0, 1],
        [2, 50, 135, 88, 210, 27.5, 0, 0],
        
        # Bad (High Risk)
        [2, 60, 148, 92, 235, 30.5, 1, 1],
        [1, 70, 158, 96, 240, 32.2, 1, 1],
        [2, 55, 142, 90, 220, 28.4, 1, 0],
    ])
    
    columns = ["Gender", "Age", "Systolic BP", "Diastolic BP", "Cholesterol", "BMI", "Smoker", "Diabetes"]
    validation_df = pd.DataFrame(validation_samples, columns=columns)
    
    return validation_df

def main():
    """
    Main function to run model training pipeline
    """
    print("=== Model Training Pipeline ===\n")
    
    # Import data transformation
    from data_transformation import main as data_main
    
    # Get processed data
    X_train, X_test, y_train, y_test, feature_names = data_main()
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Train models
    dt_model = trainer.train_decision_tree(X_train, y_train)
    rf_model = trainer.train_random_forest(X_train, y_train)
    cb_model = trainer.train_catboost(X_train, y_train)
    
    # Generate predictions
    predictions = trainer.predict_all(X_test)
    
    # Evaluate models
    results = {}
    for name, y_pred in predictions.items():
        accuracy, report = trainer.evaluate_model(name, y_test, y_pred)
        results[name] = accuracy
    
    # Save models
    trainer.save_all_models()
    
    # Create validation samples
    validation_df = create_validation_samples()
    
    # Test on validation samples
    print("\n=== Validation Sample Predictions ===")
    validation_predictions = dt_model.predict(validation_df)
    for i, pred in enumerate(validation_predictions):
        print(f"Sample {i+1} → Predicted Health Risk: {pred}")
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()