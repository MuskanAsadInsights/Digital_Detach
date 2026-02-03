"""
============================================================
THRESHOLD CALIBRATION & DECISION LOGIC
============================================================
Project: Digital Detach (FYP-1)
Objective: Calibrate the decision boundary for binary risk 
           classification and justify SMOTE implementation.
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. DATA INGESTION & AUDIT
# ============================================================
print("="*60)
print("PHASE 1: DATA DISTRIBUTION AUDIT")
print("="*60)

# Load raw dataset
df = pd.read_csv('data/raw/teen_phone_addiction_dataset.csv')
scores = df['Addiction_Level']

print(f"Total Samples: {len(df)}")
print(f"Score Range:   {scores.min()} to {scores.max()}")

# Statistical Summary
print("\nDescriptive Statistics:")
print(scores.describe())

# ============================================================
# 2. CEILING EFFECT ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PHASE 2: CEILING EFFECT & DENSITY ANALYSIS")
print("="*60)

# Calculate frequency of the maximum score
max_score_count = (scores == 10.0).sum()
max_score_pct = (max_score_count / len(df)) * 100

print(f"Maximum Score (10.0) Frequency: {max_score_count} samples")
print(f"Ceiling Effect Percentage:     {max_score_pct:.2f}%")

# Research Inference: 
# High density at the maximum score (Ceiling Effect) justifies 
# the transition from regression to binary classification.

# ============================================================
# 3. FINAL BINARY CALIBRATION
# ============================================================
print("\n" + "="*60)
print("PHASE 3: BINARY DECISION BOUNDARY")
print("="*60)

# Define Target: Severe (1) if Score is 10.0, Moderate (0) otherwise
df['Target'] = (df['Addiction_Level'] == 10.0).astype(int)

moderate_count = (df['Target'] == 0).sum()
severe_count = (df['Target'] == 1).sum()
imbalance_ratio = max(moderate_count, severe_count) / min(moderate_count, severe_count)

print(f"Moderate/At-Risk Category (0): {moderate_count} ({moderate_count/len(df):.1%})")
print(f"Severe Addiction Category (1): {severe_count} ({severe_count/len(df):.1%})")
print(f"Final Imbalance Ratio:         {imbalance_ratio:.2f}:1")



