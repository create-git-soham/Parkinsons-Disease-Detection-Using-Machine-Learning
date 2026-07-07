# 🩺 Parkinson's Disease Detection Using Machine Learning

An end-to-end machine learning project and clinical web application for evaluating Parkinson's Disease risk based on 22 acoustic voice measurement features. 

The system leverages advanced feature engineering, subject-level cross-validation to prevent data leakage, and multiple classification algorithms to provide clinicians with a reliable, interpretable probability score.

---

## 🚀 Key Features

* **Machine Learning Pipeline (`train.py`)**:
  * Evaluates multiple classifiers including Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost, and LightGBM.
  * Employs **Subject-Level Stratified Splits** & **GroupKFold Cross-Validation** to guarantee that data from the same patient is never shared between the training and validation sets.
  * Handles collinearity in voice stability features (Jitter/Shimmer) using PCA and Lasso feature selection.
* **Clinical Assessment Dashboard (`app.py`)**:
  * Modern, interactive UI built using Streamlit.
  * **Bulk Copy-Paste Input**: Clinicians can paste a spreadsheet row of 22 voice features directly to get instant risk evaluation.
  * **Manual Parameter Inputs**: Interactive sliders initialized to dataset medians for step-by-step diagnostic adjustments.
  * Color-coded indicator displaying risk category (Low, Moderate, High) with corresponding clinical interpretation notes.

---

## 📁 Repository Structure

```text
├── 01_EDA.ipynb               # Exploratory Data Analysis & visualization
├── 02_Data_Cleaning.ipynb     # Data preprocessing & feature engineering tests
├── 03_Model_Building.ipynb    # Model experimentation & prototyping
├── app.py                     # Streamlit clinical dashboard application
├── train.py                   # Model training and pipeline serialization script
├── utlis.py                   # Reusable preprocessing, helper, and evaluation code
├── parkinsons.csv             # Raw Parkinson's dataset
├── parkinsons_cleaned.csv     # Cleaned dataset
├── requirements.txt           # Project dependencies
└── models/
    └── best_model_pipeline.joblib  # Serialized best model and scaling pipeline
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/create-git-soham/Parkinsons-Disease-Detection-Using-Machine-Learning.git
   cd Parkinsons-Disease-Detection-Using-Machine-Learning
   ```

2. **Install dependencies**:
   Make sure you have Python (version 3.9+) installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   Run the training pipeline to evaluate all models and save the best performing pipeline:
   ```bash
   python train.py
   ```

4. **Launch the Dashboard**:
   Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset & Features

The dataset consists of 195 voice recordings from 31 subjects (23 with Parkinson's Disease). Each record contains **22 acoustic features**:
* **Fundamental Frequencies**: Average (`Fo_Hz`), maximum (`Fhi_Hz`), and minimum (`Flo_Hz`) vocal frequencies.
* **Jitter Metrics**: Variations in fundamental frequency (`Jitter_pct`, `Jitter_Abs`, `RAP`, `PPQ`, `Jitter_DDP`).
* **Shimmer Metrics**: Variations in amplitude (`Shimmer`, `Shimmer_dB`, `Shimmer_APQ3`, `Shimmer_APQ5`, `APQ`, `Shimmer_DDA`).
* **Noise Measures**: Harmonic-to-noise ratio (`HNR`) and noise-to-harmonic ratio (`NHR`).
* **Nonlinear Dynamics**: Recurrence period density entropy (`RPDE`), correlation dimension (`D2`), detrended fluctuation analysis (`DFA`), and pitch period entropy (`PPE`).

---

## 📈 Model Performance

The pipeline tests and ranks models using **Test F1-Score**. In the latest evaluation, the **Lasso feature-selection pipeline** coupled with the **Gradient Boosting** classifier yielded the best results:
* **Pipeline**: Lasso Feature Selection (reduced collinear features)
* **Classifier**: Gradient Boosting
* **Cross-Validation F1-Score**: 0.8372
* **Holdout Test F1-Score**: 0.9394
* **Holdout Test Accuracy**: 90.70%
* **Holdout Test Recall (Sensitivity)**: 100.00%
