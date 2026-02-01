DIGITAL DETACH (V1)
Automated Smartphone Addiction Risk Classification

Phase: FYP-1 — Final Implementation & Validation
Institute: IMSciences

PROJECT SUMMARY

Digital Detach is a hybrid AI system designed to identify and mitigate smartphone addiction risk in teens.
The FYP-1 phase establishes a scientifically validated machine-learning backbone, delivering 91% accuracy on unseen test data and 93% stability through cross-validation. This phase focuses on model reliability, robustness, and statistical justification, forming a solid foundation for advisory automation in FYP-2.

COMPLETED MILESTONES (FYP-1)

1. DATA RESEARCH & EXPLORATORY ANALYSIS (EDA)

Ceiling Effect Identification:
Discovered a dominant behavioral cluster at Score = 10.0, representing 50.8% of high-risk users, indicating saturation in addiction severity.

Target Re-definition:
Justified the transition from a clinical 3-class formulation (suffering from a 60:1 imbalance) to a Binary Classification framework (Moderate vs. Severe).

Predictor Validation:
Statistically validated usage duration and pickup frequency as reliable indicators of addictive behavior.

2. PREPROCESSING & FEATURE ENGINEERING

Modular Pipeline Design:
Implemented a reusable preprocessing pipeline (src/preprocessing.py) for data cleaning, normalization, and transformation.

Novel Behavioral Ratios:
Engineered Checking Intensity (Pickups per Hour) to capture compulsive interaction patterns more effectively than raw counts.

Robustness Enhancement:
Integrated Gaussian Noise Injection during training to simulate potential OCR variability in future Vision-API-based data extraction.

3. MODEL ARCHITECTURE & TRAINING

Algorithm Selection:
Deployed an XGBoost Ensemble model for its superior performance on imbalanced, tabular behavioral data.

Generalization Strategy:
Applied 10-Fold Stratified Cross-Validation to ensure demographic-independent stability and prevent overfitting.

4. VALIDATION & REPORTING SUITE

Automated Evaluation Engine:
Developed src/validate.py to generate all defense-grade evaluation artifacts.

Visual and Statistical Outputs:
Automated generation of ROC Curves, Confusion Matrices, and Feature Importance visualizations.

PERFORMANCE BENCHMARKS

Metric | Result | Validation Context

Mean CV Accuracy | 93.00% | 10-Fold Stratified Cross-Validation
Hold-out Test Accuracy | 91.00% | Evaluation on unseen data
AUC-ROC Score | 0.98 | Excellent class separation
Recall (Severe Class) | 0.93 | High sensitivity for risk detection

REPOSITORY STRUCTURE

Digital_Detach_V1/
|-- data/ Raw and processed datasets
|-- models/ Serialized models, scalers, encoders
|-- src/ Core modules (preprocess, train, validate)
|-- notebooks/ EDA and research justification
|-- reports/  visuals and metric reports
|-- requirements.txt Project dependencies

NEXT PHASE: FYP-2 ROADMAP

Building upon this validated foundation, FYP-2 will introduce:

Vision-Based Automation:
Integration of Gemini Vision API to extract behavioral metrics from “Screen Time” screenshots.

Hybrid Advisory Logic:
A tiered intervention framework providing educational nudges for moderate risk and structured detox recommendations for severe risk.

CONCLUSION

Digital Detach V1 delivers a production-ready, statistically validated machine-learning system for smartphone addiction risk classification.
Through 10-Fold Stratified Cross-Validation and a 93% Mean Accuracy, this phase establishes a reliable and defensible foundation for automated extraction and personalized advisory interventions in FYP-2.