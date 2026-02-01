import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score

class TrainingEngine:
    def __init__(self):
        self.input_dir = 'data/processed/'
        self.models_dir = 'models/'
        
        # Settings for XGBoost to get high accuracy
        self.model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )

    def execute(self):
        # Load the files from Preprocessing
        X_train = np.load(os.path.join(self.input_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(self.input_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(self.input_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(self.input_dir, 'y_test.npy'))

        # 10-Fold Cross-Validation (To show the panel your model is stable)
        print("Starting 10-Fold Cross-Validation...")
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Train the final model
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        print("\nFinal Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # Save the "Brain" (Model)
        joblib.dump(self.model, os.path.join(self.models_dir, 'addiction_model.pkl'))
        print(f"Model saved in {self.models_dir}")

if __name__ == "__main__":
    TrainingEngine().execute()