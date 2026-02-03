import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

class TrainingEngine:
    def __init__(self, model_type='xgboost'):
        self.input_dir = 'data/processed/'
        self.models_dir = 'models/'
        self.model_type = model_type
        
        if model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=1000, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
            )
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    def execute(self):
        # Load preprocessed data
        X_train = np.load(os.path.join(self.input_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(self.input_dir, 'y_train.npy')).ravel()
        X_test = np.load(os.path.join(self.input_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(self.input_dir, 'y_test.npy')).ravel()
        
        # 10-Fold Stratified Cross-Validation (Stability Check)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='accuracy')
        
        # Calculate Stability Metrics
        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        
        # Train and save the final model
        self.model.fit(X_train, y_train)
        
        # Decide save path based on selection
        filename = 'addiction_model.pkl' if self.model_type == 'xgboost' else 'random_forest_base.pkl'
        joblib.dump(self.model, os.path.join(self.models_dir, filename))
        
        return {"mean": mean_cv, "std": std_cv, "test_acc": accuracy_score(y_test, self.model.predict(X_test))}

if __name__ == "__main__":
    print("--- Training Pipeline Started ---")
    rf_res = TrainingEngine(model_type='rf').execute()
    xgb_res = TrainingEngine(model_type='xgboost').execute()
    
    print(f"\nRandom Forest: Mean Accuracy: {rf_res['mean']:.4f} | StdDev: {rf_res['std']:.4f}")
    print(f"XGBoost:       Mean Accuracy: {xgb_res['mean']:.4f} | StdDev: {xgb_res['std']:.4f}")