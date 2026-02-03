import joblib
import pandas as pd
import numpy as np
import os

class AddictionPredictor:
    def __init__(self, model_type='xgboost'):
        self.models_dir = 'models/'
        model_file = 'addiction_model.pkl' if model_type == 'xgboost' else 'random_forest_base.pkl'
        
        # Load artifacts
        self.model = joblib.load(os.path.join(self.models_dir, model_file))
        self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.pkl'))
        self.feature_names = joblib.load(os.path.join(self.models_dir, 'feature_names.pkl'))
        
        # Load encoders
        self.gender_le = joblib.load(os.path.join(self.models_dir, 'gender_encoder.pkl'))
        self.school_le = joblib.load(os.path.join(self.models_dir, 'school_grade_encoder.pkl'))
        self.purpose_le = joblib.load(os.path.join(self.models_dir, 'phone_usage_purpose_encoder.pkl'))

    def predict(self, age, gender, school, usage_h, sleep_h, gpa, interactions, exercise, anxiety, depression, esteem, control, bed_screen, pickups, apps, social_h, gaming_h, edu_h, purpose, family, weekend_h):
        # 1. Create a dictionary with ALL 21 raw columns (Pre-Engineering)
        data = {
            'Age': [age], 'Gender': [gender], 'School_Grade': [school],
            'Daily_Usage_Hours': [usage_h], 'Sleep_Hours': [sleep_h],
            'Academic_Performance': [gpa], 'Social_Interactions': [interactions],
            'Exercise_Hours': [exercise], 'Anxiety_Level': [anxiety],
            'Depression_Level': [depression], 'Self_Esteem': [esteem],
            'Parental_Control': [control], 'Screen_Time_Before_Bed': [bed_screen],
            'Phone_Checks_Per_Day': [pickups], 'Apps_Used_Daily': [apps],
            'Time_on_Social_Media': [social_h], 'Time_on_Gaming': [gaming_h],
            'Time_on_Education': [edu_h], 'Phone_Usage_Purpose': [purpose],
            'Family_Communication': [family], 'Weekend_Usage_Hours': [weekend_h]
        }
        df = pd.DataFrame(data)

        # 2. Replicate Behavioral Ratios (Total features becomes 24)
        df['Usage_Sleep_Ratio'] = df['Daily_Usage_Hours'] / (df['Sleep_Hours'] + 0.001)
        df['Checking_Intensity'] = df['Phone_Checks_Per_Day'] / (df['Daily_Usage_Hours'] + 0.001)
        df['Social_Media_Dominance'] = df['Time_on_Social_Media'] / (df['Daily_Usage_Hours'] + 0.001)

        # 3. Numerical Encode Text
        df['Gender'] = self.gender_le.transform(df['Gender'])
        df['School_Grade'] = self.school_le.transform(df['School_Grade'])
        df['Phone_Usage_Purpose'] = self.purpose_le.transform(df['Phone_Usage_Purpose'])
        # Handle Parental_Control as binary since it's likely 'Yes'/'No'
        df['Parental_Control'] = df['Parental_Control'].map({'Yes': 1, 'No': 0}).fillna(0)

        # 4. ALIGN & SCALE (Ensures all 24 features are in correct order)
        df_final = df[self.feature_names] 
        scaled_data = self.scaler.transform(df_final)

        # 5. Inference
        prediction = self.model.predict(scaled_data)[0]
        confidence = self.model.predict_proba(scaled_data)[0][prediction]
        
        status = "SEVERE RISK (Level 10.0)" if prediction == 1 else "MODERATE/LOW RISK"
        print(f"INPUT -> Usage: {usage_h}h | Pickups: {pickups}")
        print(f"RESULT -> {status} ({confidence:.2%} confidence)\n")

# --- EXECUTE DEMO ---
if __name__ == "__main__":
    print("--- DIGITAL DETACH: FINAL INFERENCE ENGINE ---")
    engine = AddictionPredictor(model_type='xgboost')
    
    # Example 1: Healthy Profile (16 base columns provided)
    engine.predict(17, 'Female', '11th', 2.0, 8.0, 85, 10, 2.0, 1, 1, 8, 'Yes', 0.5, 20, 5, 0.5, 0.5, 1.0, 'Education', 8, 3.0)
    
    # Example 2: Severe Profile (High Pickups, Low Sleep, High Anxiety)
    engine.predict(16, 'Male', '10th', 9.0, 4.0, 55, 2, 0.0, 8, 7, 3, 'No', 3.0, 280, 25, 5.0, 3.0, 0.0, 'Social Media', 2, 12.0)