import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self):
       
        self.data_path = 'data/raw/teen_phone_addiction_dataset.csv'
        self.output_dir = 'data/processed/'
        self.models_dir = 'models/'
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def execute(self):
        print("="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"\n[1] Loaded dataset: {df.shape}")
        
        # Define binary target
        df['Target'] = (df['Addiction_Level'] == 10.0).astype(int)
        
        print(f"[2] Target distribution:")
        print(df['Target'].value_counts())
        
        # Feature Engineering
        df['Usage_Sleep_Ratio'] = df['Daily_Usage_Hours'] / (df['Sleep_Hours'] + 0.001)
        df['Checking_Intensity'] = df['Phone_Checks_Per_Day'] / (df['Daily_Usage_Hours'] + 0.001)
        df['Social_Media_Dominance'] = df['Time_on_Social_Media'] / (df['Daily_Usage_Hours'] + 0.001)
        
        print(f"[3] Created 3 behavioral ratios")
        
        # Remove non-predictive columns
        drop_list = ['ID', 'Name', 'Location', 'Addiction_Level', 'Target']
        X = df.drop(columns=[c for c in drop_list if c in df.columns])
        y = df['Target']
        
        print(f"[4] Feature matrix shape: {X.shape}")
        print(f"    Features: {list(X.columns)}")
        
        # Noise injection
        noise_factor = 0.15
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            X[col] = X[col] + np.random.normal(0, noise_factor, X[col].shape)
        
        print(f"[5] Applied Gaussian noise (σ={noise_factor}) to {len(num_cols)} features")
        
        # Encode categoricals
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            joblib.dump(le, f'{self.models_dir}{col.lower()}_encoder.pkl')
            print(f"    Encoded: {col}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[6] Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"[7] Scaled features (Mean≈0, Std≈1)")
        print(f"    Sample scaled values: {X_train_scaled[0][:3]}")
        
        # Save preprocessing assets
        joblib.dump(scaler, f'{self.models_dir}scaler.pkl')
        joblib.dump(X.columns.tolist(), f'{self.models_dir}feature_names.pkl')
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"[8] SMOTE: {len(X_train_scaled)} → {len(X_resampled)} samples")
        
        # Save arrays
        np.save(f'{self.output_dir}X_train.npy', X_resampled)
        np.save(f'{self.output_dir}X_test.npy', X_test_scaled)
        np.save(f'{self.output_dir}y_train.npy', y_resampled)
        np.save(f'{self.output_dir}y_test.npy', y_test)
        
        print(f"\n Preprocessing complete!")
        print("="*60)

if __name__ == "__main__":
    Preprocessor().execute()