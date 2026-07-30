# Clinical Disease Diagnostics Engine

AI-assisted heart disease risk screening platform built with a Random Forest
classifier and deployed as a single-file, industry-styled Streamlit application.

IMPORTANT: This tool provides an educational risk estimate only. It is NOT a
medical diagnosis. Always consult a qualified healthcare professional for any
health concerns or before making medical decisions.


## Overview

Cardiovascular disease remains one of the leading causes of death worldwide,
and early risk identification can make a critical difference in patient
outcomes. This project uses clinical measurements — age, chest pain type,
blood pressure, cholesterol, ECG results, and more — to estimate the
probability of heart disease and flag cases that may need further medical
evaluation.

The system was trained on the UCI Heart Disease dataset (302 patient records,
after deduplication) and deployed as a full interactive dashboard, not just a
notebook model.


## Features

- Risk Prediction — enter patient clinical details through a guided form and
  receive an instant disease-probability estimate with a visual risk gauge
- Risk Classification — automatically categorizes results into Low, Moderate,
  or High risk with tailored next-step recommendations
- Explainable Results — surfaces the key contributing factors behind each
  individual prediction, combining model feature importance with how far the
  patient's values deviate from the population average
- Clinical Reference Ranges — inline guidance on normal ranges for blood
  pressure, cholesterol, and heart rate to give inputs real-world context
- Analytics Dashboard — disease distribution, age and heart-rate patterns,
  chest-pain-type breakdowns, a full feature correlation heatmap, and a model
  performance card
- Batch Prediction — upload a CSV of multiple patients and receive risk
  predictions for every row in one pass, downloadable as CSV
- Prediction History — every assessment is logged locally with trend charts
- PDF Reports — generate a downloadable, clinic-ready report for every
  prediction, including risk level, contributing factors, and a medical
  disclaimer


## Tech Stack

Python | Streamlit | scikit-learn (Random Forest Classifier) | Pandas | NumPy
| Matplotlib / Seaborn | fpdf2


## Model Performance

Evaluated on a held-out 20% test split (random_state=42):

| Metric     | Score |
|------------|-------|
| Accuracy   | 83.6% |
| Precision  | 78.8% |
| Recall     | 89.7% |
| F1-Score   | 83.9% |
| ROC-AUC    | 0.879 |

Algorithm: Random Forest Classifier, 100 estimators
Training data: 302 patient records (13 clinical features)


## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone or download the project

[    git clone https://github.com/farhan-ml/clinical-diagnostics-engine.git
](https://clinical-disease-diagnostics-engine-fumd7fhagutwvmwyrhflss.streamlit.app/)
cd clinical-diagnostics-engine

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Run the app

    streamlit run app.py

Open the URL shown in your terminal (usually http://localhost:8501).

Note: Streamlit apps must be launched with "streamlit run app.py" — running
with "python app.py" will not work correctly.


## Project Structure

    clinical-diagnostics-engine/
    ├── app.py                          Full Streamlit app (UI + ML logic + PDF reports)
    ├── random_forest.pkl               Trained RandomForestClassifier
    ├── heart_disease_scaler.pkl        StandardScaler fitted on training data
    ├── heart_disease_columns.pkl       Exact feature column order the model expects
    ├── cardiac_arrest.csv              Training dataset
    ├── requirements.txt                Python dependencies
    └── README.md


## How Risk Levels Are Determined

| Risk Level     | Probability Range | Recommendation                                  |
|----------------|--------------------|--------------------------------------------------|
| High Risk      | >= 66%             | Consult a cardiologist promptly                  |
| Moderate Risk  | 33% - 66%          | Schedule a routine check-up                      |
| Low Risk       | < 33%              | Continue regular annual check-ups                |


## Clinical Feature Reference

| Field     | Description                          | Encoding                                                              |
|-----------|---------------------------------------|-------------------------------------------------------------------------|
| age       | Age in years                          | Numeric                                                                  |
| sex       | Biological sex                        | 0 = Female, 1 = Male                                                     |
| cp        | Chest pain type                       | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
| trestbps  | Resting blood pressure (mm Hg)        | Numeric                                                                  |
| chol      | Serum cholesterol (mg/dl)             | Numeric                                                                  |
| fbs       | Fasting blood sugar > 120 mg/dl       | 0 = No, 1 = Yes                                                          |
| restecg   | Resting ECG results                   | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy  |
| thalach   | Maximum heart rate achieved           | Numeric                                                                  |
| exang     | Exercise-induced angina               | 0 = No, 1 = Yes                                                          |
| oldpeak   | ST depression induced by exercise     | Numeric                                                                  |
| slope     | Slope of the peak exercise ST segment | 0 = Upsloping, 1 = Flat, 2 = Downsloping                                 |
| ca        | Major vessels colored by fluoroscopy  | 0-4                                                                      |
| thal      | Thalassemia                           | 0 = Unknown, 1 = Fixed defect, 2 = Normal, 3 = Reversible defect         |


## Deployment

This app can be deployed for free on Streamlit Community Cloud:

1. Push this project to a GitHub repository
2. Go to share.streamlit.io and sign in with GitHub
3. Select this repository and set the main file to app.py
4. Click Deploy

Requires no server management — Streamlit Cloud installs requirements.txt
automatically.


## Testing

This app has been verified with:
- Static analysis (py_compile, pyflakes) — zero syntax or undefined-name issues
- Streamlit's AppTest framework — simulated form submissions across low-risk,
  high-risk, and min/max edge-case patient profiles, all passing without
  exceptions
- Every categorical dropdown option cycled individually to confirm correct
  encoding
- Batch-prediction logic verified against real dataset rows
- PDF report generation verified


## Disclaimer

This project is intended for educational and portfolio purposes only. The
predictions generated by this tool are not a substitute for professional
medical advice, diagnosis, or treatment. Always seek the advice of a
physician or other qualified health provider with any questions you may have
regarding a medical condition.


## License

This project is licensed under the MIT License.


## Acknowledgements

Built using the UCI Heart Disease dataset. Special thanks to Zafar Iqbal for
mentorship and guidance throughout this project.
