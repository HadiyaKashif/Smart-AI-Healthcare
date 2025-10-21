import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_ingestion import load_and_explore_data, check_data_quality

class DataCleaner:
    """
    Class for cleaning and preprocessing health data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.numerical_cols = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'BMI']
        self.categorical_cols = ['Gender', 'Smoker', 'Diabetes']
        
    def remove_unnecessary_columns(self):
        """
        Remove columns that are not needed for analysis
        """
        print("Removing unnecessary columns...")
        
        # Remove 'Name' column as it's not useful for analysis
        if 'Name' in self.df.columns:
            self.df = self.df.drop(columns='Name')
            print("✓ Removed 'Name' column")
        
        # Check correlation between weight, height and BMI to decide which to keep
        if all(col in self.df.columns for col in ['Weight (kg)', 'BMI', 'Height (cm)']):
            correlation_matrix = self.df[['Weight (kg)', 'BMI', 'Height (cm)']].corr()
            print("\nCorrelation between weight, height and BMI:")
            print(correlation_matrix)
            
            # Since BMI = Weight / Height^2, we can remove weight and height
            # BMI provides a better standardized measure
            self.df = self.df.drop(columns=['Weight (kg)', 'Height (cm)'])
            print("✓ Removed 'Weight (kg)' and 'Height (cm)' columns (BMI is sufficient)")
        
        print(f"Remaining columns: {list(self.df.columns)}")
        return self.df
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("\nHandling missing values...")
        
        missing_before = self.df.isnull().sum().sum()
        print(f"Missing values before cleaning: {missing_before}")
        
        # Fill numerical missing values with mean
        numerical_cols_with_missing = self.df.select_dtypes(include=[np.number]).columns
        self.df[numerical_cols_with_missing] = self.df[numerical_cols_with_missing].fillna(
            self.df[numerical_cols_with_missing].mean()
        )
        
        # For categorical columns, we could use mode, but let's check if there are any missing
        categorical_cols_with_missing = self.df.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols_with_missing:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        missing_after = self.df.isnull().sum().sum()
        print(f"Missing values after cleaning: {missing_after}")
        print("✓ Missing values handled")
        
        return self.df
    
    def standardize_categorical_data(self):
        """
        Standardize categorical data (convert to lowercase, strip whitespace)
        """
        print("\nStandardizing categorical data...")
        
        for col in self.categorical_cols:
            if col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].str.strip().str.lower()
                    print(f"✓ Standardized '{col}' column")
        
        # Check unique values after standardization
        for col in self.categorical_cols:
            if col in self.df.columns:
                print(f"Unique values in '{col}': {self.df[col].unique()}")
        
        return self.df
    
    def detect_and_handle_outliers(self):
        """
        Detect and handle outliers using IQR method
        """
        print("\nDetecting and handling outliers...")
        
        # Create boxplots for numerical columns before handling outliers
        self._create_boxplots("before_outlier_handling")
        
        # Handle outliers for BMI
        if 'BMI' in self.df.columns:
            Q1 = np.percentile(self.df['BMI'], 25, method='midpoint')
            Q3 = np.percentile(self.df['BMI'], 75, method='midpoint')
            IQR = Q3 - Q1
            bounds = Q1 - 1.45 * IQR, Q3 + 1.45 * IQR
            self.df['BMI'] = self.df['BMI'].clip(*bounds)
            print(f"✓ Handled outliers in 'BMI' (bounds: {bounds})")
        
        # Handle outliers for Diastolic BP
        if 'Diastolic BP' in self.df.columns:
            Q1 = np.percentile(self.df['Diastolic BP'], 25, method='midpoint')
            Q3 = np.percentile(self.df['Diastolic BP'], 75, method='midpoint')
            IQR = Q3 - Q1
            bounds = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            self.df['Diastolic BP'] = self.df['Diastolic BP'].clip(*bounds)
            print(f"✓ Handled outliers in 'Diastolic BP' (bounds: {bounds})")
        
        # Create boxplots after handling outliers
        self._create_boxplots("after_outlier_handling")
        
        return self.df
    
    def _create_boxplots(self, stage):
        """
        Create boxplots for numerical columns
        """
        for col in self.numerical_cols:
            if col in self.df.columns:
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=self.df, x=col)
                plt.title(f'Boxplot of {col} ({stage})')
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
    
    def validate_data_cleaning(self):
        """
        Validate the data cleaning process
        """
        print("\n=== DATA CLEANING VALIDATION ===")
        
        # Check for remaining missing values
        missing_after = self.df.isnull().sum().sum()
        print(f"Remaining missing values: {missing_after}")
        
        # Check data types
        print(f"\nFinal data types:\n{self.df.dtypes}")
        
        # Check dataset shape
        print(f"\nFinal dataset shape: {self.df.shape}")
        
        # Check basic statistics
        print(f"\nBasic statistics after cleaning:")
        print(self.df.describe())
        
        return self.df
    
    def get_cleaned_data(self):
        """
        Return the cleaned dataframe
        """
        return self.df

def main():
    """
    Main function to execute the data cleaning pipeline
    """
    # Load data
    file_path = r"C:\Users\Adnan\Desktop\hachakthon\Data\enhanced_health_data.csv"
    df = load_and_explore_data(file_path)
    
    if df is not None:
        # Initialize data cleaner
        cleaner = DataCleaner(df)
        
        # Execute cleaning pipeline
        cleaner.remove_unnecessary_columns()
        cleaner.handle_missing_values()
        cleaner.standardize_categorical_data()
        cleaner.detect_and_handle_outliers()
        
        # Validate and get cleaned data
        cleaned_df = cleaner.validate_data_cleaning()
        
        # Save cleaned data
        cleaned_df.to_csv("cleaned_health_data.csv", index=False)
        print("\n✓ Cleaned data saved as 'cleaned_health_data.csv'")
        
        return cleaned_df
    else:
        print("Failed to load data. Please check the file path.")
        return None

if __name__ == "__main__":
    cleaned_data = main()