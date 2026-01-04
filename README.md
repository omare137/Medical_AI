# 🏥 Medical AI

A comprehensive machine learning platform for cardiovascular disease detection and diagnosis using multiple medical imaging and signal modalities.

---

## 📋 Overview

This project develops AI models for cardiac health assessment across four major data types:

| Modality | Dataset | Task | Best Model Performance |
|----------|---------|------|------------------------|
| **12-Lead ECG** | PTB-XL (21,837 records) | Multi-label classification | Macro F1: 0.70 (SVM) |
| **ECG Signals** | MIT-BIH Arrhythmia | Arrhythmia detection | CNN-based classification |
| **Clinical Features** | Heart Disease UCI | Binary classification | F1: 0.90, AUC: 0.95 (Random Forest) |
| **Cardiac MRI** | ACDC | Segmentation + Radiomics | Deep learning pipeline |
| **Echocardiography** | EchoNet | Multi-task prediction | Video-based analysis |

---

## 🗂️ Project Structure

```
medical_ai/
├── ptb-xl/                          # PTB-XL 12-Lead ECG Analysis
│   ├── ptbxl_baseline_classification.ipynb   # ML baseline (LR, RF, SVM)
│   ├── ptbxl_deep_learning.ipynb             # Deep learning models
│   ├── ptbxl_macro_f1_optimized.ipynb        # Optimized pipeline
│   └── outputs_baseline/                      # Results & visualizations
│
├── ecg/                             # ECG Signal Processing
│   ├── cardiology.ipynb             # Main analysis notebook
│   ├── preprocess_mitbih.py         # MIT-BIH preprocessing
│   └── checkpoints/                 # Saved models & weights
│
├── ecg2.0/                          # MIT-BIH Arrhythmia Analysis
│   ├── MIT-BIH.ipynb                # Primary analysis
│   ├── MIT-BIH-DL.ipynb             # Deep learning approach
│   ├── MIT-BIH-TwoStage-Colab.ipynb # Two-stage classification
│   └── mit-bih-arrhythmia-database-1.0.0/
│
├── heart_disease/                   # Clinical Feature-Based Prediction
│   ├── heart_disease_model.ipynb    # Model training
│   ├── advanced_heart_disease_model.ipynb
│   ├── heart_disease_model.pkl      # Trained model
│   └── model_metadata.json          # Performance metrics
│
├── MRI/ACDC/                        # Cardiac MRI Analysis
│   ├── acdc_pipeline.ipynb          # Segmentation pipeline
│   ├── radiomics_classical_ml.ipynb # Radiomics features + ML
│   ├── advanced_radiomics_pipeline.ipynb
│   └── database/                    # ACDC dataset
│
├── echo/                            # Echocardiography Analysis
│   ├── echonet_multi_task_model.ipynb
│   └── preprocess_videos.py
│
└── medical_ui/                      # Web Application
    ├── backend/                     # Django REST API
    │   ├── accounts/                # User auth & roles
    │   └── requirements.txt
    └── frontend/                    # React UI
        └── src/
```

---

## 🫀 PTB-XL ECG Classification

### Target Conditions (Superclasses)

| Code | Condition | Description |
|------|-----------|-------------|
| **NORM** | Normal | Healthy ECG |
| **MI** | Myocardial Infarction | Heart attack |
| **STTC** | ST/T Changes | Ischemic changes |
| **CD** | Conduction Disturbance | Electrical pathway issues |
| **HYP** | Hypertrophy | Enlarged heart chambers |

### Pipeline

1. **Signal Processing**: 100Hz 12-lead ECG, bandpass filter (0.5-40 Hz)
2. **Feature Extraction**: 194 features per ECG
   - Statistical: mean, std, skew, kurtosis
   - Cardiac: heart rate, QRS duration
   - Frequency: FFT components
3. **Multi-Label Classification**: One-vs-Rest strategy
4. **Cross-Validation**: Stratified 5-Fold CV

### Results

| Model | Macro F1 | Micro F1 |
|-------|----------|----------|
| Logistic Regression | 0.659 | — |
| Random Forest | 0.667 | — |
| **SVM (RBF)** | **0.695** | — |

---

## ❤️ Heart Disease Prediction

Binary classification using clinical features from the UCI Heart Disease dataset.

### Features

- **Demographics**: Age, Sex
- **Vitals**: Resting BP, Max Heart Rate
- **Lab Values**: Cholesterol, Fasting Blood Sugar
- **ECG Findings**: Resting ECG, ST Slope
- **Symptoms**: Chest Pain Type, Exercise Angina

### Performance

| Metric | Score |
|--------|-------|
| Accuracy | 89.9% |
| Precision | 94.3% |
| Recall | 86.8% |
| F1 Score | 90.4% |
| ROC AUC | 0.95 |

---

## 🖼️ Cardiac MRI (ACDC)

Analysis of the Automated Cardiac Diagnosis Challenge dataset for:
- Left/Right ventricle segmentation
- Radiomics feature extraction
- Cardiac function assessment

---

## 🌐 Web Application

Role-based medical platform with JWT authentication.

### User Roles

| Role | Description |
|------|-------------|
| **GP** | General Practitioner |
| **SPECIALIST** | Cardiologist, etc. |
| **RECEPTIONIST** | Administrative staff |
| **ADMIN** | System administrator |

### Tech Stack

- **Backend**: Django 5.0, Django REST Framework, JWT Auth
- **Frontend**: React 18, React Router
- **Database**: SQLite (dev) / PostgreSQL (prod)

### Quick Start

```bash
# Backend
cd medical_ui/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Frontend
cd medical_ui/frontend
npm install
npm start
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- 16GB+ RAM recommended for deep learning

### Python Dependencies

```bash
# Core ML
pip install numpy pandas scikit-learn scipy

# Deep Learning
pip install tensorflow keras torch

# ECG Processing
pip install wfdb

# Visualization
pip install matplotlib seaborn

# Web App
pip install django djangorestframework djangorestframework-simplejwt
```

### Data Setup

1. **PTB-XL**: Download from [PhysioNet](https://physionet.org/content/ptb-xl/)
2. **MIT-BIH**: Download from [PhysioNet](https://physionet.org/content/mitdb/)
3. **ACDC**: Request from [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

---

## 📊 Key Notebooks

| Notebook | Description |
|----------|-------------|
| `ptb-xl/ptbxl_baseline_classification.ipynb` | Complete ML pipeline for ECG classification |
| `ecg2.0/MIT-BIH-DL.ipynb` | Deep learning for arrhythmia detection |
| `heart_disease/advanced_heart_disease_model.ipynb` | Clinical risk prediction |
| `MRI/ACDC/acdc_pipeline.ipynb` | Cardiac MRI segmentation |

---

## 🔬 Methods

### Signal Processing
- Butterworth bandpass filtering
- R-peak detection for heart rate
- QRS complex analysis

### Machine Learning
- Logistic Regression, Random Forest, SVM
- One-vs-Rest for multi-label problems
- Stratified K-Fold cross-validation
- Class balancing with sample weights

### Deep Learning
- 1D CNNs for ECG classification
- ResNet architectures
- Multi-task learning for echo analysis

---

## 📈 Future Work

- [ ] Transformer models for ECG (ECG-BERT)
- [ ] Explainability (SHAP, Grad-CAM)
- [ ] Multi-modal fusion (ECG + Clinical + Imaging)
- [ ] Real-time inference API
- [ ] FDA-style validation pipeline
- [ ] External dataset validation

---

## 📚 References

1. Wagner, P., et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*.
2. Moody, G.B., & Mark, R.G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMBS*.
3. Bernard, O., et al. (2018). Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation. *IEEE TMI*.

---

## 📄 License

This project is for research and educational purposes. Medical AI models require clinical validation before any real-world use.

---

## 👥 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

<p align="center">
  <i>Built with ❤️ for advancing cardiovascular health</i>
</p>

