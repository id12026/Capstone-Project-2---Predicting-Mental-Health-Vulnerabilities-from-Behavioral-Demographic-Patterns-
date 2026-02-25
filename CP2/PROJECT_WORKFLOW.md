# Mental Health Vulnerability Prediction System - Project Workflow

## Project Overview

This is a comprehensive mental health vulnerability prediction system that uses hybrid machine learning (Autoencoder + XGBoost) to predict mental health risk levels (Low/Medium/High) from behavioral and demographic patterns.

## Complete Workflow & Architecture

### 1. Data Collection & Preprocessing Phase

```
Raw Data Input (mental_health_data.csv)
    ↓
Data Loading & Validation
    ↓
Feature Engineering
    ├── Behavioral Features (7): stress, mood_swings, coping_struggles, work_interest, social_weakness, changes_habits, family_history
    ├── Demographic Features (5): gender, occupation, country, days_indoors, age
    └── Label Encoding & Scaling
        ├── Label Encoders for categorical variables
        ├── StandardScaler for numerical features
        └── Save preprocessors (models/preprocessors/)
    ↓
Train/Test Split (80/20)
```

### 2. Model Training Phase

#### 2.1 Autoencoder Training (Deep Learning Component)
```
Behavioral Features (10 dims) → Autoencoder
    ↓
Encoder Architecture:
    Input(10) → Dense(48, ReLU) → Dropout(0.3)
    → Dense(24, ReLU) → Dropout(0.3)
    → Dense(12, ReLU) → Dropout(0.3)
    → Dense(3, ReLU) [Latent Space]
    ↓
Decoder Architecture:
    Latent(3) → Dense(12, ReLU) → Dropout(0.3)
    → Dense(24, ReLU) → Dropout(0.3)
    → Dense(48, ReLU) → Dropout(0.3)
    → Dense(10, Sigmoid) [Reconstruction]
    ↓
Training Parameters:
    ├── Loss: MSE (Mean Squared Error)
    ├── Optimizer: Adam (lr=0.001)
    ├── Regularization: L2 (0.005)
    ├── Early Stopping: patience=12
    └── Output: Latent Features (3 dims)
```

#### 2.2 Hybrid Model Training (Machine Learning Component)
```
Feature Combination:
    ├── Latent Features from Autoencoder (3 dims)
    ├── Demographic Features (5 dims)
    └── Combined Features (8 dims)
    ↓
XGBoost Classifier Training:
    ├── Input: Combined Features (8 dims)
    ├── Classes: 3 (Low/Medium/High Risk)
    ├── Objective: multi:softprob
    ├── Regularization: Balanced
    └── Target Accuracy: 84-86%
```

### 3. Model Deployment Phase

#### 3.1 Flask Web Application
```
Flask Server (app.py)
    ├── Routes:
    │   ├── GET / - Main assessment form
    │   ├── POST /predict - Risk prediction API
    │   ├── POST /shap_explanation - Explainability API
    │   ├── GET /dashboard - Admin analytics
    │   ├── GET /analytics - System metrics
    │   ├── POST /feedback - User feedback
    │   └── GET /health - Health check
    ↓
Request Processing:
    ├── Input Validation
    ├── Feature Preprocessing (using saved encoders/scalers)
    ├── Model Prediction
    ├── Result Formatting
    └── Database Storage (SQLite)
```

#### 3.2 Prediction Pipeline
```
User Input (Web Form)
    ↓
Data Preparation:
    ├── Behavioral Feature Mapping (Low=0, Medium=1, High=2)
    ├── Demographic Feature Encoding
    ├── Feature Scaling
    └── Shape Validation
    ↓
Model Inference:
    ├── Autoencoder Feature Extraction
    ├── XGBoost Classification
    ├── Probability Calculation
    └── Risk Level Assignment
    ↓
Output Generation:
    ├── Risk Level (Low/Medium/High)
    ├── Confidence Score
    ├── Class Probabilities
    ├── Behavioral Score
    ├── Top Contributing Factors
    └── Personalized Recommendations
```

### 4. Explainability Phase

#### 4.1 SHAP Integration
```
SHAP Explanation Pipeline:
    ├── TreeExplainer Initialization (XGBoost model)
    ├── SHAP Values Calculation
    ├── Multi-class Handling (3 risk levels)
    ├── Feature Importance Ranking
    └── Safe Scalar Conversion (Fixed Issue)
    ↓
Output:
    ├── Top 5 Contributing Features
    ├── Feature Impact Direction (+/-)
    ├── Visual Explanations (plots)
    └── User-friendly Descriptions
```

## File Structure & Responsibilities

### Core Application Files
- **`app.py`** - Main Flask application with all API endpoints
- **`model_training.py`** - Hybrid model architecture and training logic
- **`preprocessing.py`** - Data loading, cleaning, and preprocessing
- **`main.py`** - Application entry point with CLI arguments
- **`explainability.py`** - SHAP-based model explainability

### Supporting Files
- **`utils/logger.py`** - Enhanced logging system
- **`requirements.txt`** - Python dependencies
- **`train_with_regularization.py`** - Training with regularization techniques
- **`diagnose_issue.py`** - System diagnostic utilities
- **`quick_fix.py`** - Troubleshooting and fixes

### Data & Model Storage
- **`data/`** - Dataset storage
- **`models/`** - Trained models and preprocessors
- **`logs/`** - Application logs
- **`static/`** - Web assets (CSS, JS, images)
- **`templates/`** - HTML templates

## Technology Stack

### Backend Technologies
- **Flask 2.3.3** - Web framework
- **TensorFlow 2.13.0** - Deep learning (Autoencoder)
- **XGBoost 1.7.6** - Gradient boosting classifier
- **Scikit-learn 1.3.0** - ML utilities and preprocessing
- **SHAP 0.42.1** - Model explainability

### Data Processing
- **Pandas 2.0.3** - Data manipulation
- **NumPy 1.24.3** - Numerical computing
- **Joblib 1.3.1** - Model serialization

### Visualization & Frontend
- **Matplotlib 3.7.2** - Plotting
- **Seaborn 0.12.2** - Statistical visualization
- **HTML/CSS/JavaScript** - Web interface
- **Bootstrap 5.1.3** - UI framework

### Database & Storage
- **SQLite** - Lightweight database for predictions and analytics
- **HDF5 (.h5)** - Neural network model storage
- **Pickle (.pkl)** - Scikit-learn model storage

## Key Features & Innovations

### 1. Hybrid Intelligence Architecture
- **Autoencoder**: Learns behavioral patterns and reduces dimensionality
- **XGBoost**: Provides accurate classification with feature importance
- **Combined**: Leverages strengths of both deep learning and traditional ML

### 2. Prevention-First Approach
- **Risk Levels**: Low/Medium/High (not binary diagnosis)
- **Early Intervention**: Identifies vulnerability before clinical stage
- **Actionable Insights**: Provides recommendations based on risk factors

### 3. Explainable AI
- **SHAP Values**: Explains feature contributions
- **Visual Explanations**: Charts and plots for understanding
- **Transparent**: Users understand why they received certain risk level

### 4. Privacy-Preserving Design
- **No Personal Identifiers**: Anonymous predictions
- **Local Processing**: Models run locally, not cloud
- **Data Protection**: Ethical and compliant design

## Performance & Validation

### Model Performance
- **Target Accuracy**: 84-86% (balanced against overfitting)
- **Regularization**: L2 (0.005) + Dropout (30%)
- **Cross-validation**: Train/Val/Test splits
- **Early Stopping**: Prevents overtraining

### System Monitoring
- **Health Checks**: `/health` endpoint
- **Performance Metrics**: Response times, error rates
- **User Analytics**: Dashboard for insights
- **Logging**: Comprehensive error tracking

## Deployment & Usage

### Development Setup
```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run application
python main.py --mode serve --port 5000
```

### Production Deployment
- **Web Server**: Flask development server (for demo)
- **Database**: SQLite (can be upgraded to PostgreSQL)
- **Static Files**: Served by Flask
- **Models**: Pre-trained and loaded at startup

## Quality Assurance & Testing

### Error Handling
- **Input Validation**: Form data validation
- **Graceful Degradation**: Fallback mechanisms
- **Comprehensive Logging**: Error tracking and debugging
- **User Feedback**: Continuous improvement loop

### Model Robustness
- **Feature Padding**: Handles missing features
- **Shape Validation**: Ensures correct input dimensions
- **Fallback Predictions**: Default values when models fail
- **Regularization**: Prevents overfitting

## Future Enhancements

### Technical Improvements
- **Real-time Monitoring**: Live performance metrics
- **Model Versioning**: A/B testing capabilities
- **API Rate Limiting**: Prevent abuse
- **Database Scaling**: Support for larger datasets

### Feature Enhancements
- **Time Series Analysis**: Track changes over time
- **Multi-language Support**: International accessibility
- **Mobile Application**: Native mobile experience
- **Integration APIs**: Connect with health systems

This comprehensive workflow demonstrates a production-ready mental health prediction system with robust architecture, explainable AI, and user-centric design.
