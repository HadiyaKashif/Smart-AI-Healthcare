import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.metrics_file = "evaluation_metrics.json"
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def load_test_data(self):
        """Load test data from CSV files"""
        try:
            X_test = pd.read_csv("X_test.csv")
            y_test = pd.read_csv("y_test.csv").squeeze()  # Convert to Series
            feature_names = X_test.columns.tolist()
            
            print(f"Test data loaded: {X_test.shape}")
            print(f"Target data loaded: {y_test.shape}")
            
            return X_test, y_test, feature_names
        except Exception as e:
            print(f"Error loading test data: {e}")
            return None, None, None
    
    def comprehensive_evaluation(self, model, X_test, y_test, model_name):
        """Perform comprehensive model evaluation"""
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class names
        classes = list(y_test.unique())
        
        # Store results with better structure
        self.results[model_name] = {
            'model_info': {
                'name': model_name,
                'evaluation_timestamp': self.timestamp,
                'test_set_size': len(X_test)
            },
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class_metrics': {
                'classes': classes,
                'precision': [float(x) for x in precision_per_class],
                'recall': [float(x) for x in recall_per_class],
                'f1_score': [float(x) for x in f1_per_class]
            },
            'detailed_report': report,
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'classes': classes
            }
        }
        
        return accuracy, report, cm
    
    def save_metrics_to_json(self):
        """Save all evaluation metrics to JSON file"""
        try:
            # Prepare comprehensive JSON structure
            json_output = {
                'evaluation_metadata': {
                    'timestamp': self.timestamp,
                    'total_models_evaluated': len(self.results),
                    'models': list(self.results.keys())
                },
                'model_comparison': self._create_comparison_data(),
                'detailed_results': self.results
            }
            
            # Save to JSON file
            with open(self.metrics_file, 'w') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            
            print(f"Evaluation metrics saved to {self.metrics_file}")
            
        except Exception as e:
            print(f"Error saving metrics to JSON: {e}")
    
    def _create_comparison_data(self):
        """Create comparison data for JSON output"""
        comparison = {}
        for model_name, result in self.results.items():
            comparison[model_name] = {
                'accuracy': result['overall_metrics']['accuracy'],
                'precision': result['overall_metrics']['precision'],
                'recall': result['overall_metrics']['recall'],
                'f1_score': result['overall_metrics']['f1_score'],
                'rank': None
            }
        
        # Add ranking based on accuracy
        ranked_models = sorted(comparison.items(), 
                             key=lambda x: x[1]['accuracy'], 
                             reverse=True)
        
        for rank, (model_name, _) in enumerate(ranked_models, 1):
            comparison[model_name]['rank'] = rank
        
        return comparison
    
    def print_model_comparison(self):
        """Print model comparison in console"""
        if not self.results:
            print("No models evaluated yet!")
            return
        
        print("\n=== Model Comparison ===")
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{result['overall_metrics']['accuracy']:.3f}",
                'Precision': f"{result['overall_metrics']['precision']:.3f}",
                'Recall': f"{result['overall_metrics']['recall']:.3f}",
                'F1-Score': f"{result['overall_metrics']['f1_score']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        print(comparison_df.to_string(index=False))
        
        return comparison_df

def main():
    """
    Main function to run model evaluation pipeline
    """
    print("=== Model Evaluation Pipeline ===\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load test data from CSV files
    X_test, y_test, feature_names = evaluator.load_test_data()
    
    if X_test is None or y_test is None:
        print("Error: Could not load test data. Please run data_transformation.py and model_training.py first.")
        return None, None
    
    # Load trained models
    models_to_evaluate = {
        'decision_tree': 'decision_tree_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'catboost': 'catboost_model.pkl'
    }
    
    # Evaluate each model
    for model_name, model_path in models_to_evaluate.items():
        print(f"Evaluating {model_name}...")
        model = evaluator.load_model(model_path)
        
        if model is not None:
            accuracy, report, cm = evaluator.comprehensive_evaluation(
                model, X_test, y_test, model_name
            )
            print(f"  Accuracy: {accuracy:.3f}")
    
    # Save metrics to JSON file
    evaluator.save_metrics_to_json()
    
    # Print model comparison
    comparison_df = evaluator.print_model_comparison()
    
    print(f"\nEvaluation completed. Results saved to {evaluator.metrics_file}")
    
    return evaluator, comparison_df

if __name__ == "__main__":
    evaluator, comparison_df = main()