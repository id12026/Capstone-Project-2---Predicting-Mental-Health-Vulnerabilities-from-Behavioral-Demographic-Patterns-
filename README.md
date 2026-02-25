# Predicting Mental Health Vulnerabilities from Behavioral & Demographic Patterns

## By:
- Mohitha Bandi (22WUO0105037)
- Pailla Bhavya (22WUO0105020)
- T. Harshavardhan Reddy (22WUO0105023)
- Y. Siddhartha Reddy (22WU0105028)
- Pragnan Seemakurthi (22WU0105021)

## Supervised By:
Dr. Resham Raj Shivwanshi  
Assistant Professor, Woxsen University

## Table of Contents
- [Introduction](#introduction)
- [Research Problem](#research-problem)
- [Motivation & Real-World Importance](#motivation--real-world-importance)
- [Related Work](#related-work)
- [Technologies and Methodologies](#technologies-and-methodologies)
- [Proposed Approach & Architecture](#proposed-approach--architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [API Endpoints](#api-endpoints)
- [Results](#results)
- [Research Gaps and Challenges](#research-gaps-and-challenges)
- [Scope](#scope)
- [Expected Outcomes](#expected-outcomes)

## Introduction

Mental health issues are rapidly increasing due to stress, isolation, and lifestyle imbalance. Many individuals do not seek timely help due to stigma and lack of awareness. Machine Learning (ML) and Deep Learning (DL) enable early detection of mental health vulnerability through non-invasive data analysis.

This project predicts mental health risk using behavioral indicators such as:
- Mood swings
- Work interest changes
- Social weaknesses
- Coping struggles
- Stress patterns

The system aims to support early intervention and promote proactive mental well-being.

## Research Problem

Despite growing awareness, mental health disorders often go undetected until they reach a critical stage. Current diagnostic methods rely heavily on self-reporting, clinical interviews, or observable symptoms, which are reactive, subjective, and inaccessible to many.

There is no scalable, preventive system that uses everyday behavioral and demographic data to predict vulnerability—not just diagnosis—in the general population.

## Motivation & Real-World Importance

### Motivation
- Mental health issues are increasing globally, especially after COVID-19
- Early identification can:
  - Prevent severe conditions
  - Enable timely support
  - Reduce social stigma
- Using everyday behavioral and demographic data makes prediction scalable

### Real-World Importance
- Helps organizations design employee wellness programs
- Assists healthcare providers in prioritizing high-risk individuals
- Supports public health planning and awareness campaigns
- Allows individuals to assess risk anonymously

## Related Work

### Existing Approaches
- Clinical tools: PHQ-9, GAD-7 – require active participation
- Machine learning in mental health: Mostly based on social media text (sentiment, linguistic cues) for depression detection
- Traditional surveys: Large-scale studies (e.g., WHO surveys) identify risk factors but do not provide real-time individual predictions

## Technologies and Methodologies

### Technologies
- **Backend**: Flask web framework
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript

### Architecture Components
- **Autoencoder**: Neural network for behavioral feature extraction
- **XGBoost Classifier**: Gradient boosting for final risk classification
- **Flask API**: RESTful web service for predictions
- **SQLite Database**: Storage for predictions and analytics

### Limitations
- Most models predict diagnosis rather than vulnerability (pre-clinical stage)
- Lack integration of demographic, behavioral, and environmental factors
- Not designed for low-resource or non-clinical settings

## Proposed Approach & Architecture

### Novelty

#### Prevention-First Framework
- Predicts vulnerability, not diagnosis
- Risk classification: Low / Medium / High
- Enables early intervention

#### Hybrid Intelligence Architecture
- Autoencoders for behavior pattern learning
- Combination of ML and DL models
- Transfer learning for adaptability

#### Explainable AI
- Highlights important features
- Explains why a person is at risk
- Provides actionable insights

#### Privacy-Preserving Design
- No personal identifiers collected
- Fully anonymous predictions
- Ethical and data-protection compliant

## Project Structure
```
├── app.py                    # Flask web application
├── model_training.py         # ML model training pipeline
├── preprocessing.py          # Data preprocessing module
├── train_with_regularization.py  # Training with regularization
├── main.py                   # Main application entry point
├── explainability.py         # SHAP-based explainability
├── diagnose_issue.py         # Diagnostic utilities
├── quick_fix.py              # Quick fixes and troubleshooting
├── utils/
│   └── logger.py            # Enhanced logging utilities
├── models/                   # Trained model storage
├── data/                     # Dataset directory
├── logs/                     # Application logs
├── static/                   # Static assets (CSS, JS)
├── templates/                # HTML templates
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-vulnerability-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. Start the Flask server:
```bash
python main.py --mode serve --port 5000
```

2. Access the web interface at: `http://localhost:5000`

3. For training the model:
```bash
python main.py --mode train
```

4. For testing data loading:
```bash
python main.py --mode test
```

### Training the Model
```bash
python train_with_regularization.py --epochs 50 --batch-size 32
```

## Model Architecture

### Hybrid Model Structure
1. **Autoencoder Neural Network**:
   - Input: Behavioral features (10 dimensions)
   - Encoder: Dense(48) → Dropout(30%) → Dense(24) → Dropout(30%) → Dense(12) → Dense(3)
   - Decoder: Dense(12) → Dropout(30%) → Dense(24) → Dropout(30%) → Dense(48) → Dense(output_dim)
   - Purpose: Dimensionality reduction and feature extraction

2. **XGBoost Classifier**:
   - Input: Combined features (autoencoder output + demographic features)
   - Classes: 3 risk levels (Low, Medium, High)
   - Hyperparameters: Balanced regularization for target accuracy (84-86%)

### Target Accuracy
- Configured for 84-86% accuracy to prevent overfitting
- Balanced regularization strategy implemented

## API Endpoints

### Main Endpoints
- `GET /` - Main assessment form
- `POST /predict` - Mental health risk prediction API
- `GET /test_predict` - Test endpoint with default values
- `GET /predict/info` - API information
- `GET /dashboard` - Admin analytics dashboard
- `GET /analytics` - System performance metrics
- `POST /feedback` - User feedback submission
- `GET /health` - Health check endpoint

### Prediction Request Format
```json
{
  "gender": "Female",
  "age": 30,
  "occupation": "Corporate",
  "country": "United States",
  "family_history": "No",
  "days_indoors": "1-14 days",
  "growing_stress": "Maybe",
  "mood_swings": "Medium",
  "coping_struggles": "Sometimes",
  "work_interest": "Maybe",
  "social_weakness": "Maybe",
  "changes_habits": "Sometimes"
}
```

### Prediction Response Format
```json
{
  "success": true,
  "risk_level": "Low Risk",
  "confidence": 85.23,
  "probabilities": {
    "Low Risk": 85.23,
    "Medium Risk": 12.45,
    "High Risk": 2.32
  },
  "behavioral_score": 8,
  "top_contributing_factors": ["Growing Stress (Score: 2)", "Mood Swings (Score: 2)"],
  "recommendations": ["Maintain a regular sleep schedule", "Practice mindfulness or meditation"]
}
```

## Results
<img width="1925" height="805" alt="image" src="https://github.com/user-attachments/assets/683259bd-c562-42be-a520-15f365db1922" />
<img width="798" height="939" alt="image" src="https://github.com/user-attachments/assets/2f66abbc-cd87-45f8-b9fd-042882cfaf2c" />
<img width="879" height="1125" alt="image" src="https://github.com/user-attachments/assets/2a77a4e6-5f4b-4aab-b438-9cccff021bbb" />
<img width="1405" height="939" alt="image" src="https://github.com/user-attachments/assets/16391d81-16d6-4312-af5b-6574149f3988" />
<img width="943" height="386" alt="image" src="https://github.com/user-attachments/assets/abfca03d-d6de-424b-a19e-caa58fa0743a" />
<img width="958" height="194" alt="image" src="https://github.com/user-attachments/assets/4a6f4638-1255-4305-9346-941274493de3" />
<img width="958" height="455" alt="image" src="https://github.com/user-attachments/assets/0934ec0c-01b9-4219-9419-325b77565e2e" />
<img width="2000" height="1037" alt="image" src="https://github.com/user-attachments/assets/9fee5182-4286-4c55-ac5a-ef0862a8d596" />
<img width="570" height="388" alt="image" src="https://github.com/user-attachments/assets/be01942a-7489-45f0-ac3b-a104592a6eba" />
<img width="1023" height="497" alt="image" src="https://github.com/user-attachments/assets/bb0c8014-f180-4bb7-9146-b99af33fb561" />
<img width="1023" height="150" alt="image" src="https://github.com/user-attachments/assets/2e2ccfea-0484-482c-991e-df76f7e1113f" />
<img width="977" height="437" alt="image" src="https://github.com/user-attachments/assets/80636735-7227-4f61-8728-af45e944713a" />
<img width="1984" height="478" alt="image" src="https://github.com/user-attachments/assets/70b85868-f04d-4396-8e75-8e3d88ffa83d" />


### Model Performance
- **Target Accuracy**: 84-86% (to prevent overfitting)
- **Classification**: 3-tier risk assessment (Low/Medium/High)
- **Generalization**: Balanced regularization prevents overfitting while maintaining performance
- **Interpretability**: Feature importance and contributing factors provided

### Risk Categories
1. **Low Risk**: Minimal behavioral indicators of mental health vulnerability
2. **Medium Risk**: Some concerning patterns requiring monitoring
3. **High Risk**: Significant indicators requiring immediate attention

## Research Gaps and Challenges

### Gaps
- No unified model using demographic + behavioral + lifestyle data
- Absence of graded risk score (Low / Medium / High) instead of binary classification
- Scarcity of systems usable outside clinical environments with minimal user input

### Challenges
- Data privacy and ethical use
- Self-reported data bias due to stigma
- Imbalanced datasets
- Generalizability across cultures, occupations, and genders
- Dynamic nature of mental health (temporal patterns)

## Scope

- Prediction of mental health vulnerability level (Low / Medium / High)
- Hybrid ML + DL model for improved accuracy
- Explainable AI to identify significant risk factors
- Early-warning alert system
- Web-based decision-support platform for individuals, students, and workplaces

## Expected Outcomes

- Accurate mental health vulnerability prediction
- Early warning alerts for at-risk individuals
- Improved awareness and timely psychological support
- Benefits students, employees, and remote populations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the project team members listed above.

## Acknowledgments

- Dr. Resham Raj Shivwanshi for supervision and guidance
- Mental health professionals who provided insights
- Open-source community for the tools and libraries used
