import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path=r"C:\Users\Adnan\Desktop\hachakthon\Data\cleaned_health_data.csv"):
    """
    Load and preprocess the health dataset
    """
    # Load data
    df = pd.read_csv(file_path)
    
    print("Dataset Info:")
    df.info()
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def encode_features(X, save_to_csv=True, filename="processed_data.csv"):
    """
    Encode categorical features and optionally save processed data to CSV
    """
    # Create a copy to avoid modifying the original
    X_encoded = X.copy()
    
    # Encode Gender
    X_encoded['Gender'] = X_encoded['Gender'].map({'male': 1, 'female': 0})
    
    # Convert boolean columns to integers
    bool_columns = ['Smoker', 'Diabetes']
    for col in bool_columns:
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].astype(int)
    
    # Save to CSV if requested
    if save_to_csv:
        try:
            X_encoded.to_csv(filename, index=False)
            print(f"Processed data saved to {filename}")
            print(f"Processed data shape: {X_encoded.shape}")
            print(f"Processed data columns: {list(X_encoded.columns)}")
            print("\nFirst few rows of processed data:")
            print(X_encoded.head())
        except Exception as e:
            print(f"Error saving processed data to CSV: {e}")
    
    return X_encoded

def prepare_features_target(df, target_column='Health', save_processed_data=True):
    """
    Prepare features and target variables using raw columns directly
    """
    # Separate features and target - use all columns except the target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables and save processed data
    X = encode_features(X, save_to_csv=save_processed_data, filename="processed_features.csv")
    
    # Also save the target variable separately
    if save_processed_data:
        try:
            y.to_csv("processed_target.csv", index=False)
            print(f"Target data saved to processed_target.csv")
            print(f"Target distribution:\n{y.value_counts()}")
        except Exception as e:
            print(f"Error saving target data to CSV: {e}")
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42, save_splits=True):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save train/test splits if requested
    if save_splits:
        try:
            X_train.to_csv("X_train.csv", index=False)
            X_test.to_csv("X_test.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)
            y_test.to_csv("y_test.csv", index=False)
            print("Train/test splits saved to CSV files")
        except Exception as e:
            print(f"Error saving train/test splits: {e}")
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_complete_processed_dataset(df, filename="complete_processed_dataset.csv"):
    """
    Save the complete processed dataset with all transformations
    """
    try:
        df_processed = df.copy()
        
        # Apply the same encoding to the complete dataset
        df_processed['Gender'] = df_processed['Gender'].map({'male': 1, 'female': 0})
        df_processed['Smoker'] = df_processed['Smoker'].astype(int)
        df_processed['Diabetes'] = df_processed['Diabetes'].astype(int)
        
        df_processed.to_csv(filename, index=False)
        print(f"Complete processed dataset saved to {filename}")
        print(f"Dataset shape: {df_processed.shape}")
        print("\nFirst few rows of complete processed dataset:")
        print(df_processed.head())
        
    except Exception as e:
        print(f"Error saving complete processed dataset: {e}")

def main():
    """
    Main function to run data transformation pipeline
    """
    print("=== Data Transformation Pipeline ===\n")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Save complete processed dataset
    save_complete_processed_dataset(df)
    
    # Prepare features and target using raw columns directly
    X, y = prepare_features_target(df, save_processed_data=True)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, save_splits=True)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = main()