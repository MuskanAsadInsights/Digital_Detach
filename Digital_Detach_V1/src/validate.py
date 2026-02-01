"""
============================================================
MODEL VALIDATION &  METRICS GENERATOR
============================================================
Project: Digital Detox (FYP-1)
Objective: Generate statistical validation metrics, ROC 
           curves, and performance reports for academic 
           evaluation and defense.
============================================================
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
import os

class ModelValidator:
    def __init__(self):
        self.models_dir = 'models/'
        self.data_dir = 'data/processed/'
        self.output_dir = 'reports/'
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_assets(self):
        print("\n" + "="*60)
        print("SECTION 1: LOADING SAVED MODEL & TEST DATA")
        print("="*60)
        # Load the trained model and testing sets
        self.model = joblib.load(f'{self.models_dir}addiction_model.pkl')
        self.X_test = np.load(f'{self.data_dir}X_test.npy')
        self.y_test = np.load(f'{self.data_dir}y_test.npy')
        self.feature_names = joblib.load(f'{self.models_dir}feature_names.pkl')
        print(f"Data assets loaded. Samples in test set: {len(self.X_test)}")
        
    def generate_and_save_metrics(self):
        print("\n" + "="*60)
        print("SECTION 2: PERFORMANCE METRICS & STATISTICAL REPORTS")
        print("="*60)
        
        # Perform predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate standard classification scores
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        # Generate detailed classification report
        report = classification_report(
            self.y_test, 
            self.y_pred, 
            target_names=['Moderate/At-Risk (0)', 'Severe Addiction (1)']
        )
        
        # Output to terminal console
        print(f"Global Accuracy: {self.accuracy:.4f}")
        print(f"AUC-ROC Score:  {self.auc:.4f}\n")
        print("Detailed Statistical Report:")
        print(report)
        
        # Export comprehensive metrics to text file
        report_path = os.path.join(self.output_dir, 'validation_report.txt')
        with open(report_path, 'w') as f:
            f.write("DIGITAL DETOX MODEL VALIDATION REPORT\n")
            f.write("="*40 + "\n")
            f.write(f"Accuracy: {self.accuracy:.4f}\n")
            f.write(f"AUC-ROC Score: {self.auc:.4f}\n\n")
            f.write("Classification Matrix Data:\n")
            f.write(str(confusion_matrix(self.y_test, self.y_pred)) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Detailed metrics successfully saved to {report_path}")

    def plot_visuals(self):
        print("\n" + "="*60)
        print("SECTION 3: GENERATING ANALYTICAL VISUALIZATIONS")
        print("="*60)
        
        # 1. Confusion Matrix Generation
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Moderate', 'Severe'], 
                    yticklabels=['Moderate', 'Severe'],
                    annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        plt.title('Confusion Matrix: Prediction Performance', fontsize=15, pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}confusion_matrix.png', dpi=300)
        print(f"Visualization saved: confusion_matrix.png")

        # 2. ROC Curve Generation
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#e74c3c', linewidth=3, 
                 label=f'XGBoost (AUC = {self.auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', 
                 label='Baseline (Random)')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=15)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}roc_curve.png', dpi=300)
        print(f"Visualization saved: roc_curve.png")

        # 3. Feature Importance 
        importances = self.model.feature_importances_
        feat_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        top_10 = feat_df.sort_values('Importance', ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_10, 
                    hue='Feature', palette='viridis', legend=False)
        plt.title('Top 10 Behavioral Predictors', fontsize=16, fontweight='bold', pad=20)
        plt.subplots_adjust(left=0.35)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: feature_importance.png")

    def execute(self):
        self.load_assets()
        self.generate_and_save_metrics()
        self.plot_visuals()
        print("\n" + "="*60)
        print("VALIDATION PROCESS COMPLETE")
        print("="*60)

if __name__ == "__main__":
    validator = ModelValidator()
    validator.execute()