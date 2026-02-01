import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self):
        # Paths to your files
        self.data_path = 'data/raw/teen_phone_addiction_dataset.csv'

'
        self.output_dir = 'data/processed/'
        self.models_dir = 'models/'
        
        # Create folders if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def execute(self):
        # Load the original CSV
        df = pd.read_csv(self.data_path)
        
        # Define the Target (What we want to predict)
        df['Target'] = (df['Addiction_Level'] == 10.0).astype(int)
        
        # FEATURE ENGINEERING: Create the 'Ratios' to help accuracy
        df['Usage_Sleep_Ratio'] = df['Daily_Usage_Hours'] / (df['Sleep_Hours'] + 1)
        df['Checking_Intensity'] = df['Phone_Checks_Per_Day'] / (df['Daily_Usage_Hours'] + 1)
        df['Social_Media_Dominance'] = df['Time_on_Social_Media'] / (df['Daily_Usage_Hours'] + 1)

        # Remove columns that the model shouldn't see (Prevent "Cheating")
        drop_list = ['ID', 'Name', 'Location', 'User_ID', 'Addiction_Level', 'Target']
        X = df.drop(columns=[c for c in drop_list if c in df.columns]).dropna()
        y = df['Target']

        # Add small Noise (0.15) to help model handle Gemini Vision errors later
        noise_factor = 0.15 
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            X[col] = X[col] + np.random.normal(0, noise_factor, X[col].shape)

        # Convert text columns to numbers
        cat_cols = X.select_dtypes(include=['object', 'str']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            joblib.dump(le, f'{self.models_dir}{col.lower()}_encoder.pkl')

        # Split data into 80% Training and 20% Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the numbers (Mean=0, Variance=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        joblib.dump(scaler, f'{self.models_dir}scaler.pkl')
        joblib.dump(X.columns.tolist(), f'{self.models_dir}feature_names.pkl')

        # Balance the data using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Save these for the Train script
        np.save(f'{self.output_dir}X_train.npy', X_resampled)
        np.save(f'{self.output_dir}X_test.npy', X_test_scaled)
        np.save(f'{self.output_dir}y_train.npy', y_resampled)
        np.save(f'{self.output_dir}y_test.npy', y_test)
        
        print("Preprocessing complete for XGBoost.")

if __name__ == "__main__":
    Preprocessor().execute()