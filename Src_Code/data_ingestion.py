import pandas as pd
import numpy as np

def load_and_explore_data(file_path):
    """
    Load the health dataset and perform initial exploration
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataframe
    """
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

def check_data_quality(df):
    """
    Perform initial data quality checks
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    dict: Data quality summary
    """
    
    quality_report = {
        'missing_values': df.isnull().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes,
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Missing values:\n{quality_report['missing_values']}")
    print(f"\nDuplicate rows: {quality_report['duplicate_rows']}")
    print(f"\nData types:\n{quality_report['data_types']}")
    print(f"\nUnique values per column:")
    for col, count in quality_report['unique_values'].items():
        print(f"  {col}: {count}")
    
    return quality_report

def main():
    """
    Main function to execute data ingestion process
    """
    file_path = r"C:\Users\Adnan\Desktop\hachakthon\Data\enhanced_health_data.csv"
    
    try:
        # Load and explore data
        df = load_and_explore_data(file_path)
        
        # Check data quality
        quality_report = check_data_quality(df)
        
        return df, quality_report
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error during data ingestion: {str(e)}")
        return None, None

if __name__ == "__main__":
    df, quality_report = main()