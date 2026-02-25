# Mental Health Vulnerability Prediction System - Flowchart

## System Architecture Flowchart

```mermaid
graph TD
    A[User Access] --> B[Web Interface]
    B --> C{Assessment Form}
    C --> D[Behavioral Data]
    C --> E[Demographic Data]
    
    D --> F[Feature Processing]
    E --> F
    F --> G[Data Validation]
    G --> H[Feature Encoding]
    H --> I[Feature Scaling]
    
    I --> J[Prediction Pipeline]
    J --> K[Autoencoder Feature Extraction]
    K --> L[Latent Features 3D]
    L --> M[Feature Combination]
    M --> N[XGBoost Classification]
    
    N --> O[Risk Prediction]
    O --> P[Low/Medium/High Risk]
    P --> Q[Confidence Score]
    Q --> R[Probability Distribution]
    R --> S[Top Contributing Factors]
    S --> T[Personalized Recommendations]
    
    T --> U[Results Display]
    U --> V[SHAP Explanation Request]
    V --> W[Explainability Engine]
    W --> X[SHAP Values Calculation]
    X --> Y[Feature Importance Ranking]
    Y --> Z[Visual Explanations]
    Z --> AA[Enhanced Results]
    
    AA --> BB[Database Storage]
    BB --> CC[Analytics Dashboard]
    CC --> DD[System Monitoring]
    
    style A fill:#e1f5fe
    style J fill:#f3e5f5
    style O fill:#e8f5e8
    style U fill:#fff3e0
    style W fill:#fce4ec
    style BB fill:#f1f8e9
```

## Data Processing Flowchart

```mermaid
graph LR
    A[Raw CSV Data] --> B[Data Loading]
    B --> C[Format Detection]
    C --> D[Column Validation]
    D --> E[Missing Value Handling]
    E --> F[Feature Engineering]
    
    F --> G[Behavioral Features]
    F --> H[Demographic Features]
    
    G --> I[Label Encoding]
    H --> J[Standard Scaling]
    
    I --> K[Train/Test Split]
    J --> K
    K --> L[Model Training]
    
    L --> M[Autoencoder Training]
    L --> N[XGBoost Training]
    
    M --> O[Latent Feature Extraction]
    N --> P[Combined Features]
    O --> P
    P --> Q[Model Evaluation]
    Q --> R[Model Deployment]
    
    style A fill:#ffebee
    style F fill:#e8f5e8
    style L fill:#e3f2fd
    style Q fill:#fff3e0
    style R fill:#f1f8e9
```

## Model Training Flowchart

```mermaid
graph TD
    A[Training Data] --> B[Data Preprocessing]
    B --> C[Feature Split]
    C --> D[Behavioral Features]
    C --> E[Demographic Features]
    
    D --> F[Autoencoder Input]
    F --> G[Encoder Network]
    G --> H[Dense 48 + ReLU]
    H --> I[Dropout 0.3]
    I --> J[Dense 24 + ReLU]
    J --> K[Dropout 0.3]
    K --> L[Dense 12 + ReLU]
    L --> M[Dropout 0.3]
    M --> N[Latent Space 3D]
    
    N --> O[Decoder Network]
    O --> P[Dense 12 + ReLU]
    P --> Q[Dropout 0.3]
    Q --> R[Dense 24 + ReLU]
    R --> S[Dropout 0.3]
    S --> T[Dense 48 + ReLU]
    T --> U[Dropout 0.3]
    U --> V[Output Dense 10 + Sigmoid]
    
    V --> W[Reconstruction Loss]
    W --> X[Backpropagation]
    X --> Y[Weight Updates]
    Y --> Z[Trained Encoder]
    
    Z --> AA[Feature Extraction]
    AA --> BB[Latent + Demographic]
    BB --> CC[XGBoost Input]
    CC --> DD[XGBoost Training]
    DD --> EE[Gradient Boosting]
    EE --> FF[Tree Ensemble]
    FF --> GG[Trained Classifier]
    
    style A fill:#e3f2fd
    style G fill:#f3e5f5
    style N fill:#e8f5e8
    style DD fill:#fff3e0
    style GG fill:#f1f8e9
```

## Prediction Request Flowchart

```mermaid
sequenceDiagram
    participant User
    participant WebUI
    participant Flask
    participant Preprocessor
    participant Autoencoder
    participant XGBoost
    participant SHAP
    participant Database
    
    User->>WebUI: Fill Assessment Form
    WebUI->>Flask: POST /predict
    Flask->>Preprocessor: Prepare Input Data
    Preprocessor->>Preprocessor: Encode & Scale Features
    Preprocessor->>Flask: Processed Features
    Flask->>Autoencoder: Extract Features
    Autoencoder->>Flask: Latent Features
    Flask->>XGBoost: Predict Risk
    XGBoost->>Flask: Risk Level + Probabilities
    Flask->>Flask: Generate Recommendations
    Flask->>Database: Store Prediction
    Flask->>WebUI: JSON Response
    WebUI->>User: Display Results
    
    User->>WebUI: Click SHAP Explanation
    WebUI->>Flask: POST /shap_explanation
    Flask->>SHAP: Calculate SHAP Values
    SHAP->>Flask: Feature Contributions
    Flask->>WebUI: Enhanced Response
    WebUI->>User: Show Explanations
```

## Error Handling Flowchart

```mermaid
graph TD
    A[Request Received] --> B{Input Validation}
    B -->|Valid| C[Process Request]
    B -->|Invalid| D[Return Error Response]
    
    C --> E{Model Loading}
    E -->|Success| F[Feature Processing]
    E -->|Failed| G[Fallback Mode]
    
    F --> H{Prediction Success}
    H -->|Success| I[Generate Results]
    H -->|Failed| J[Default Prediction]
    
    I --> K{SHAP Calculation}
    K -->|Success| L[SHAP Explanation]
    K -->|Failed| M[Dummy SHAP Values]
    
    G --> N[Basic Risk Assessment]
    J --> N
    L --> O[Complete Response]
    M --> O
    N --> O
    
    O --> P[Log Results]
    P --> Q[Send Response]
    
    D --> R[Log Error]
    Q --> S[Database Storage]
    R --> S
    
    style A fill:#e3f2fd
    style C fill:#e8f5e8
    style I fill:#fff3e0
    style O fill:#f1f8e9
    style D fill:#ffebee
    style R fill:#ffebee
```

## System Components Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[HTML5 Interface]
        B[Bootstrap CSS]
        C[JavaScript Client]
        D[Chart.js Visualizations]
    end
    
    subgraph "Application Layer"
        E[Flask Web Server]
        F[API Endpoints]
        G[Request Handlers]
        H[Response Formatters]
    end
    
    subgraph "Business Logic Layer"
        I[Data Preprocessor]
        J[Feature Engineer]
        K[Prediction Engine]
        L[Recommendation Generator]
        M[Explainability Module]
    end
    
    subgraph "Model Layer"
        N[Autoencoder NN]
        O[XGBoost Classifier]
        P[SHAP Explainer]
        Q[Feature Encoders]
        R[Data Scalers]
    end
    
    subgraph "Data Layer"
        S[SQLite Database]
        T[Model Files .h5/.pkl]
        U[Configuration Files]
        V[Log Files]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> N
    J --> N
    K --> O
    L --> O
    M --> P
    
    N --> Q
    O --> Q
    P --> Q
    Q --> R
    
    I --> S
    K --> S
    E --> V
    
    N --> T
    O --> T
    P --> T
    Q --> T
    R --> T
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
    style N fill:#fff3e0
    style S fill:#f1f8e9
```

## Key Workflow Insights

### 1. **Data Flow Architecture**
- **Input**: User behavioral and demographic data
- **Processing**: Feature encoding, scaling, and validation
- **Prediction**: Hybrid model (Autoencoder + XGBoost)
- **Output**: Risk level, confidence, explanations, recommendations

### 2. **Model Integration Strategy**
- **Autoencoder**: Extracts latent behavioral patterns (10D → 3D)
- **XGBoost**: Classifies combined features (3D + 5D demographic)
- **SHAP**: Provides explainability for predictions

### 3. **Error Resilience**
- **Input Validation**: Prevents malformed data
- **Fallback Mechanisms**: Graceful degradation when models fail
- **Comprehensive Logging**: Tracks all operations for debugging

### 4. **User Experience Flow**
- **Simple Form**: Easy data entry
- **Instant Results**: Real-time prediction
- **Detailed Explanations**: SHAP-based feature importance
- **Actionable Insights**: Personalized recommendations

This flowchart represents a production-ready mental health prediction system with robust architecture, comprehensive error handling, and user-centric design.
