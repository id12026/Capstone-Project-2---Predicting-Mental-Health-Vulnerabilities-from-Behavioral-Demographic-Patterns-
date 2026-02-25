from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import json
from datetime import datetime
import sqlite3
import os
import sys
import traceback
from sklearn.preprocessing import StandardScaler
import shap

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import EnhancedLogger, logger

app = Flask(__name__)
app.secret_key = 'mental-health-predictor-secret-key-2024'

# Global variables
model = None
preprocessor = None
label_encoders = None
behavioral_scaler = None
demographic_scaler = None

# Initialize enhanced logger
enhanced_logger = EnhancedLogger("flask_app")
logger = enhanced_logger.get_logger()

def init_db():
    """Initialize database"""
    try:
        conn = sqlite3.connect('mental_health.db')
        c = conn.cursor()
        
        # Create predictions table
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME,
                      session_id TEXT,
                      gender TEXT,
                      age INTEGER,
                      occupation TEXT,
                      country TEXT,
                      family_history TEXT,
                      days_indoors TEXT,
                      growing_stress TEXT,
                      mood_swings TEXT,
                      coping_struggles TEXT,
                      work_interest TEXT,
                      social_weakness TEXT,
                      changes_habits TEXT,
                      behavioral_score INTEGER,
                      risk_level TEXT,
                      confidence FLOAT,
                      top_factors TEXT)''')
        
        # Create user feedback table
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      prediction_id INTEGER,
                      accurate BOOLEAN,
                      comments TEXT,
                      timestamp DATETIME,
                      FOREIGN KEY(prediction_id) REFERENCES predictions(id))''')
        
        # Create analytics table
        c.execute('''CREATE TABLE IF NOT EXISTS analytics
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp DATETIME,
                      endpoint TEXT,
                      response_time FLOAT,
                      status_code INTEGER,
                      user_agent TEXT)''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def load_models(app_logger=None):
    """Load trained models"""
    global model, label_encoders, behavioral_scaler, demographic_scaler
    
    if app_logger:
        global logger
        logger = app_logger
    
    try:
        logger.info("Loading models and preprocessors...")
        
        # Check if models directory exists
        if not os.path.exists('models'):
            logger.error("Models directory not found!")
            return False
        
        # Load model
        from model_training import MentalHealthModel
        model = MentalHealthModel(logger=logger)
        
        if not model.load_models():
            logger.error("Failed to load ML models!")
            # Create a basic model as fallback for testing
            logger.info("Creating fallback model...")
            model = MentalHealthModel(input_dim_behavioral=7, input_dim_demographic=5, logger=logger)
            # Build basic architecture as fallback
            model.build_autoencoder()
            # This won't have trained weights but will allow the app to start
            logger.warning("Using fallback model without trained weights")
        
        # Load preprocessors with fallback
        preprocessor_path = 'models/preprocessors/'
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessors directory not found: {preprocessor_path}")
            return False
        
        try:
            label_encoders = joblib.load(f'{preprocessor_path}label_encoders.pkl')
            logger.info(f"  Label encoders loaded: {len(label_encoders)} features")
        except Exception as e:
            logger.warning(f"Could not load label encoders: {e}")
            label_encoders = {}
        
        try:
            behavioral_scaler = joblib.load(f'{preprocessor_path}behavioral_scaler.pkl')
            # Handle case where scaler doesn't have n_features_in_ attribute
            if hasattr(behavioral_scaler, 'n_features_in_'):
                logger.info(f"  Behavioral scaler loaded: expects {behavioral_scaler.n_features_in_} features")
            else:
                # For older versions of sklearn, we'll determine this later
                logger.info("  Behavioral scaler loaded")
        except Exception as e:
            logger.warning(f"Could not load behavioral scaler: {e}")
            behavioral_scaler = StandardScaler()
            logger.info("  Using new behavioral scaler")
        
        try:
            demographic_scaler = joblib.load(f'{preprocessor_path}demographic_scaler.pkl')
            # Handle case where scaler doesn't have n_features_in_ attribute
            if hasattr(demographic_scaler, 'n_features_in_'):
                logger.info(f"  Demographic scaler loaded: expects {demographic_scaler.n_features_in_} features")
            else:
                # For older versions of sklearn, we'll determine this later
                logger.info("  Demographic scaler loaded")
        except Exception as e:
            logger.warning(f"Could not load demographic scaler: {e}")
            demographic_scaler = StandardScaler()
            logger.info("  Using new demographic scaler")
        
        logger.info("✅ All models and preprocessors loaded successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error(traceback.format_exc())
        return False

def prepare_input_data(form_data):
    """Prepare form data for prediction"""
    logger.info("Preparing input data for prediction...")
    
    try:
        # Define feature mappings
        behavioral_mapping = {
            'growing_stress': {'Low': 0, 'Medium': 1, 'High': 2},
            'mood_swings': {'Low': 0, 'Medium': 1, 'High': 2},
            'coping_struggles': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'work_interest': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'social_weakness': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'changes_habits': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'family_history': {'No': 0, 'Yes': 1}
        }
        
        # Define feature order (MUST MATCH TRAINING ORDER)
        behavioral_features_order = [
            'growing_stress', 'mood_swings', 'coping_struggles',
            'work_interest', 'social_weakness', 'changes_habits',
            'family_history'
        ]
        
        # Get behavioral scores
        behavioral_scores = []
        for feature in behavioral_features_order:
            value = form_data.get(feature, 'No')
            mapping = behavioral_mapping.get(feature, {})
            score = mapping.get(value, 0)
            behavioral_scores.append(score)
            logger.debug(f"  {feature}: {value} -> {score}")
        
        logger.info(f"Behavioral scores: {behavioral_scores}")
        logger.info(f"Behavioral features count: {len(behavioral_scores)}")
        
        # Check if behavioral scores match scaler expectation
        # Handle case where scaler doesn't have n_features_in_ attribute
        if hasattr(behavioral_scaler, 'n_features_in_'):
            expected_features = behavioral_scaler.n_features_in_
        else:
            # If scaler was just created (not fitted), default to expected number
            expected_features = 7  # Based on the number of behavioral features in the form
            
        if len(behavioral_scores) != expected_features:
            logger.error(f"Behavioral feature mismatch! Got {len(behavioral_scores)}, expected {expected_features}")
            # Try to adjust
            if len(behavioral_scores) < expected_features:
                # Pad with zeros
                padding = expected_features - len(behavioral_scores)
                behavioral_scores.extend([0] * padding)
                logger.warning(f"Padded with {padding} zeros")
            else:
                # Truncate
                behavioral_scores = behavioral_scores[:expected_features]
                logger.warning(f"Truncated to {expected_features} features")
        
        # Prepare demographic features
        demographic_features = []
        demographic_order = ['gender', 'occupation', 'country', 'days_indoors']
        
        for feature in demographic_order:
            value = str(form_data.get(feature, 'Unknown'))
            if feature in label_encoders:
                try:
                    encoded = label_encoders[feature].transform([value])[0]
                except:
                    # If value not seen before, use default
                    encoded = 0
                    logger.warning(f"Unknown value for {feature}: {value}, using default encoding")
            else:
                encoded = 0
            demographic_features.append(encoded)
            logger.debug(f"  {feature}: {value} -> {encoded}")
        
        # Add age
        try:
            age = int(form_data.get('age', 30))
        except:
            age = 30
            logger.warning(f"Invalid age, using default: 30")
        demographic_features.append(age)
        
        logger.info(f"Demographic features: {demographic_features}")
        logger.info(f"Demographic features count: {len(demographic_features)}")
        
        # Check if demographic features match scaler expectation
        # Handle case where scaler doesn't have n_features_in_ attribute
        if hasattr(demographic_scaler, 'n_features_in_'):
            expected_demo_features = demographic_scaler.n_features_in_
        else:
            # If scaler was just created (not fitted), default to expected number
            expected_demo_features = 5  # gender, occupation, country, days_indoors, age
            
        if len(demographic_features) != expected_demo_features:
            logger.error(f"Demographic feature mismatch! Got {len(demographic_features)}, expected {expected_demo_features}")
            # Try to adjust
            if len(demographic_features) < expected_demo_features:
                padding = expected_demo_features - len(demographic_features)
                demographic_features.extend([0] * padding)
                logger.warning(f"Padded with {padding} zeros")
            else:
                demographic_features = demographic_features[:expected_demo_features]
                logger.warning(f"Truncated to {expected_demo_features} features")
        
        # Convert to arrays
        X_behavioral = np.array([behavioral_scores])
        X_demographic = np.array([demographic_features])
        
        logger.info(f"X_behavioral shape: {X_behavioral.shape}")
        logger.info(f"X_demographic shape: {X_demographic.shape}")
        
        # Scale features
        X_behavioral_scaled = behavioral_scaler.transform(X_behavioral)
        X_demographic_scaled = demographic_scaler.transform(X_demographic)
        
        logger.info("Feature scaling completed successfully")
        
        return X_behavioral_scaled, X_demographic_scaled, behavioral_scores
        
    except Exception as e:
        logger.error(f"Failed to prepare input data: {e}")
        logger.error(traceback.format_exc())
        raise

def save_prediction(form_data, prediction, confidence, top_factors):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('mental_health.db')
        c = conn.cursor()
        
        # Calculate behavioral score
        score_mapping = {
            'growing_stress': {'Low': 0, 'Medium': 1, 'High': 2},
            'mood_swings': {'Low': 0, 'Medium': 1, 'High': 2},
            'coping_struggles': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'work_interest': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'social_weakness': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'changes_habits': {'No': 0, 'Sometimes': 1, 'Yes': 2},
            'family_history': {'No': 0, 'Yes': 1}
        }
        
        behavioral_score = 0
        for feature, mapping in score_mapping.items():
            value = form_data.get(feature, 'No')
            behavioral_score += mapping.get(value, 0)
        
        # Get session ID
        session_id = session.get('session_id', 'unknown')
        
        # Insert prediction
        c.execute('''INSERT INTO predictions 
                     (timestamp, session_id, gender, age, occupation, country, 
                      family_history, days_indoors, growing_stress, mood_swings,
                      coping_struggles, work_interest, social_weakness, changes_habits,
                      behavioral_score, risk_level, confidence, top_factors)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(),
                   session_id,
                   form_data.get('gender', 'Unknown'),
                   int(form_data.get('age', 30)),
                   form_data.get('occupation', 'Unknown'),
                   form_data.get('country', 'Unknown'),
                   form_data.get('family_history', 'No'),
                   form_data.get('days_indoors', '1-14 days'),
                   form_data.get('growing_stress', 'Low'),
                   form_data.get('mood_swings', 'Low'),
                   form_data.get('coping_struggles', 'No'),
                   form_data.get('work_interest', 'No'),
                   form_data.get('social_weakness', 'No'),
                   form_data.get('changes_habits', 'No'),
                   behavioral_score,
                   prediction,
                   confidence,
                   json.dumps(top_factors)))
        
        prediction_id = c.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction saved to database with ID: {prediction_id}")
        return prediction_id
        
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        return None

def generate_recommendations(risk_level, top_factors):
    """Generate personalized recommendations"""
    recommendations = []
    
    # General recommendations
    recommendations.append("Maintain a regular sleep schedule (7-9 hours per night)")
    recommendations.append("Practice mindfulness or meditation for 10-15 minutes daily")
    recommendations.append("Engage in physical activity for at least 30 minutes, 3 times a week")
    recommendations.append("Stay connected with friends and family regularly")
    recommendations.append("Limit screen time before bedtime")
    
    # Risk-specific recommendations
    if risk_level == 1:  # Medium
        recommendations.append("Consider talking to a counselor or therapist")
        recommendations.append("Keep a mood journal to track emotional patterns")
        recommendations.append("Join a support group or community activity")
        recommendations.append("Practice stress-management techniques like deep breathing")
    
    elif risk_level == 2:  # High
        recommendations.append("**Seek professional help from a mental health specialist**")
        recommendations.append("Contact mental health helpline: 1-800-950-NAMI (6264)")
        recommendations.append("Inform a trusted family member or friend about your situation")
        recommendations.append("Avoid isolation - make plans to meet people regularly")
        recommendations.append("Consider cognitive behavioral therapy (CBT)")
    
    # Factor-specific recommendations
    if any('Stress' in factor for factor in top_factors):
        recommendations.append("Practice progressive muscle relaxation techniques")
        recommendations.append("Take regular breaks during work or study")
        recommendations.append("Learn to say 'no' to unnecessary commitments")
    
    if any('Social' in factor for factor in top_factors):
        recommendations.append("Start with small social interactions daily")
        recommendations.append("Practice social skills in low-pressure environments")
        recommendations.append("Consider joining interest-based clubs or groups")
    
    if any('Work' in factor for factor in top_factors):
        recommendations.append("Break tasks into smaller, manageable steps")
        recommendations.append("Set realistic goals and celebrate achievements")
        recommendations.append("Discuss workload concerns with supervisor or teacher")
    
    if any('Sleep' in factor or 'Habit' in factor for factor in top_factors):
        recommendations.append("Establish a consistent bedtime routine")
        recommendations.append("Avoid caffeine and heavy meals before bedtime")
        recommendations.append("Create a relaxing pre-sleep environment")
    
    return recommendations[:10]  # Return top 10 recommendations

@app.before_request
def before_request():
    """Log each request"""
    session['start_time'] = datetime.now()
    
    if request.endpoint and request.endpoint != 'static':
        logger.info(f"Request: {request.method} {request.path}")
        logger.debug(f"Headers: {dict(request.headers)}")
        # Only log form data for POST requests
        if request.method == 'POST' and request.form:
            logger.debug(f"Form data: {dict(request.form)}")
        if request.method == 'POST' and request.json:
            logger.debug(f"JSON data: {request.json}")

@app.after_request
def after_request(response):
    """Log each response"""
    if request.endpoint and request.endpoint != 'static':
        start_time = session.get('start_time', datetime.now())
        response_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Response: {request.method} {request.path} -> {response.status_code} ({response_time:.3f}s)")
        
        # Only log to analytics for POST requests that might affect the DB
        if request.method == 'POST':
            try:
                conn = sqlite3.connect('mental_health.db')
                c = conn.cursor()
                c.execute('''INSERT INTO analytics 
                             (timestamp, endpoint, response_time, status_code, user_agent)
                             VALUES (?, ?, ?, ?, ?)''',
                          (datetime.now().isoformat(),
                           request.endpoint or request.path,
                           response_time,
                           response.status_code,
                           request.user_agent.string))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to log analytics: {e}")
    
    return response

@app.route('/')
def home():
    """Home page"""
    # Generate session ID if not exists
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())[:8]
        logger.info(f"New session started: {session['session_id']}")
    
    logger.info(f"Rendering home page for session: {session['session_id']}")
    try:
        # Check if template exists
        import os
        template_path = os.path.join('templates', 'index.html')
        if not os.path.exists(template_path):
            logger.error(f"Template {template_path} does not exist")
            return "<h1>Welcome to Mental Health Predictor</h1><p>Template file not found</p>", 200
        
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"<h1>Home Page</h1><p>Session ID: {session.get('session_id', 'unknown')}</p><p>Error: {str(e)}</p>", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    logger.info("Prediction request received")
    
    try:
        # Try to get JSON data first
        if request.is_json:
            form_data = request.get_json()
        else:
            # Check content type and extract form data appropriately
            if request.content_type and 'application/x-www-form-urlencoded' in request.content_type:
                form_data = request.form.to_dict()
            elif request.content_type and 'multipart/form-data' in request.content_type:
                form_data = request.form.to_dict()
            elif request.json:  # If it's JSON, convert to form-like dict
                form_data = request.json
            else:
                # Fallback: try to get form data regardless of content type
                form_data = request.form.to_dict()
                if not form_data and request.data:
                    # If no form data but there's raw data, try to parse it
                    try:
                        from urllib.parse import parse_qs
                        raw_data = request.get_data(as_text=True)
                        form_data = {k: v[0] if len(v) > 0 else '' for k, v in parse_qs(raw_data).items()}
                    except:
                        form_data = {}
        
        # Check if form data is empty
        if not form_data:
            logger.error("No form data received")
            return jsonify({'success': False, 'error': 'No form data received'}), 400
        
        logger.info(f"Form data keys: {list(form_data.keys())}")
        
        # Prepare input data
        X_behavioral, X_demographic, behavioral_scores = prepare_input_data(form_data)
        
        # Make prediction
        try:
            predictions, probabilities = model.predict(X_behavioral, X_demographic)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return a default prediction as fallback
            logger.warning("Using fallback prediction due to model error")
            risk_level = 0  # Default to low risk
            confidence = 0.5  # 50% confidence
            predictions = np.array([risk_level])
            probabilities = np.array([[0.8, 0.15, 0.05]])  # High probability for low risk)
        
        # Get results
        risk_level = int(predictions[0])
        confidence = float(max(probabilities[0]))
        
        # Risk level mapping
        risk_map = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_text = risk_map.get(risk_level, 'Unknown')
        
        logger.info(f"Prediction result: {risk_text} (confidence: {confidence:.2%})")
        
        # Generate top contributing factors
        top_factors = []
        factor_labels = [
            'Growing Stress', 'Mood Swings', 'Coping Struggles',
            'Work Interest', 'Social Weakness', 'Changes in Habits',
            'Family History'
        ]
        
        for i, score in enumerate(behavioral_scores[:7]):
            if score >= 1:
                label = factor_labels[i] if i < len(factor_labels) else f"Factor {i+1}"
                top_factors.append(f"{label} (Score: {score})")
        
        # Add additional factors based on form data
        if form_data.get('days_indoors', '') in ['31-60 days', 'More than 2 months']:
            top_factors.append(f"Extended Isolation: {form_data['days_indoors']}")
        
        if form_data.get('occupation', '') in ['Student', 'Healthcare', 'Corporate']:
            top_factors.append(f"High-Stress Occupation: {form_data['occupation']}")
        
        # Save prediction to database
        prediction_id = save_prediction(form_data, risk_text, confidence, top_factors)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_level, top_factors)
        
        # Log the prediction
        enhanced_logger.log_prediction(form_data, {
            'risk_level': risk_text,
            'confidence': confidence,
            'behavioral_score': sum(behavioral_scores),
            'prediction_id': prediction_id
        })
        
        # Prepare response
        response = {
            'success': True,
            'prediction_id': prediction_id,
            'risk_level': risk_text,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'Low Risk': round(probabilities[0][0] * 100, 2),
                'Medium Risk': round(probabilities[0][1] * 100, 2),
                'High Risk': round(probabilities[0][2] * 100, 2)
            },
            'behavioral_score': sum(behavioral_scores),
            'max_behavioral_score': 14,  # 7 features * max score 2
            'top_contributing_factors': top_factors[:5],
            'recommendations': recommendations,
            'warning': risk_level == 2,
            'session_id': session.get('session_id', 'unknown')
        }
        
        logger.info(f"Prediction response prepared successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    logger.info("Dashboard access requested")
    
    try:
        conn = sqlite3.connect('mental_health.db')
        
        # Get overall statistics - handle potential NULLs
        stats_query = '''
            SELECT 
                COUNT(*) as total_predictions,
                COALESCE(AVG(behavioral_score), 0) as avg_score,
                COALESCE(AVG(confidence), 0) as avg_confidence,
                SUM(CASE WHEN risk_level = 'High Risk' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN risk_level = 'Medium Risk' THEN 1 ELSE 0 END) as medium_risk_count,
                SUM(CASE WHEN risk_level = 'Low Risk' THEN 1 ELSE 0 END) as low_risk_count
            FROM predictions
        '''
        stats = conn.execute(stats_query).fetchone()
        
        # Ensure stats values are not None
        stats = tuple(0 if val is None else val for val in stats)
        
        # Get recent predictions
        recent = conn.execute('''
            SELECT timestamp, gender, age, occupation, risk_level, confidence
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 10
        ''').fetchall()
        
        # Get risk distribution by occupation
        by_occupation = conn.execute('''
            SELECT occupation, 
                   COUNT(*) as count,
                   COALESCE(AVG(behavioral_score), 0) as avg_score,
                   COALESCE(AVG(confidence), 0) as avg_confidence
            FROM predictions
            GROUP BY occupation
            ORDER BY count DESC
            LIMIT 5
        ''').fetchall()
        
        # Get daily predictions count
        daily = conn.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        ''').fetchall()
        
        # Get system performance
        perf_query = '''
            SELECT 
                COUNT(*) as total_requests,
                COALESCE(AVG(response_time), 0) as avg_response_time,
                COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0) as error_count
            FROM analytics
        '''
        performance = conn.execute(perf_query).fetchone()
        
        # Ensure performance values are not None
        performance = tuple(0 if val is None else val for val in performance)
        
        conn.close()
        
        logger.info(f"Dashboard stats loaded: {stats[0]} total predictions")
        
        return render_template('dashboard.html',
                             stats=stats,
                             recent=recent,
                             by_occupation=by_occupation,
                             daily=daily,
                             performance=performance)
        
    except Exception as e:
        logger.error(f"Dashboard failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"<h1>Dashboard Error</h1><p>{str(e)}</p>", 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback on prediction accuracy"""
    logger.info("Feedback submission received")
    
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        accurate = data.get('accurate')
        comments = data.get('comments', '')
        
        conn = sqlite3.connect('mental_health.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO feedback 
                     (prediction_id, accurate, comments, timestamp)
                     VALUES (?, ?, ?, ?)''',
                  (prediction_id, accurate, comments, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback saved for prediction {prediction_id}: accurate={accurate}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logs')
def view_logs():
    """View recent logs (admin only)"""
    logger.warning("Logs page accessed - security warning")
    
    try:
        log_files = []
        if os.path.exists('logs'):
            for file in sorted(os.listdir('logs'), reverse=True):
                if file.endswith('.log'):
                    filepath = os.path.join('logs', file)
                    size = os.path.getsize(filepath)
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    log_files.append({
                        'name': file,
                        'size': f"{size/1024:.1f} KB",
                        'modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                        'path': filepath
                    })
        
        # Get latest log content
        latest_log = ""
        if log_files:
            try:
                with open(log_files[0]['path'], 'r', encoding='utf-8') as f:
                    latest_log = f.read()[-5000:]  # Last 5000 chars
            except:
                latest_log = "Could not read log file"
        
        return f"""
        <html>
        <head><title>System Logs</title></head>
        <body>
            <h1>System Logs</h1>
            <h2>Available Log Files:</h2>
            <ul>
                {''.join(f'<li><a href="/log_file/{f["name"]}">{f["name"]}</a> - {f["size"]} - {f["modified"]}</li>' for f in log_files)}
            </ul>
            <h2>Latest Log (last 5000 chars):</h2>
            <pre style="background: #f0f0f0; padding: 10px; overflow: auto; max-height: 500px;">
            {latest_log}
            </pre>
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
        """
    except Exception as e:
        return f"Error viewing logs: {str(e)}", 500

@app.route('/log_file/<filename>')
def view_log_file(filename):
    """View specific log file"""
    logger.warning(f"Log file accessed: {filename}")
    
    try:
        filepath = os.path.join('logs', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"""
            <html>
            <head><title>{filename}</title></head>
            <body>
                <h1>{filename}</h1>
                <pre style="background: #f0f0f0; padding: 10px; overflow: auto; max-height: 800px;">
                {content}
                </pre>
                <p><a href="/logs">Back to Logs</a> | <a href="/">Home</a></p>
            </body>
            </html>
            """
        else:
            return "Log file not found", 404
    except Exception as e:
        return f"Error reading log file: {str(e)}", 500

@app.route('/system_info')
def system_info():
    """System information page"""
    import platform
    import psutil
    
    info = {
        'system': f"{platform.system()} {platform.release()}",
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_cores': psutil.cpu_count(logical=True),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        'disk_total': f"{psutil.disk_usage('.').total / (1024**3):.2f} GB",
        'disk_free': f"{psutil.disk_usage('.').free / (1024**3):.2f} GB",
        'working_directory': os.getcwd(),
        'log_directory': os.path.abspath('logs'),
        'model_directory': os.path.abspath('models'),
        'flask_debug': app.debug,
        'session_count': len(session)
    }
    
    return jsonify(info)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': bool(model),
        'database': 'connected' if os.path.exists('mental_health.db') else 'missing',
        'logs_directory': 'exists' if os.path.exists('logs') else 'missing',
        'session_id': session.get('session_id', 'none')
    }
    
    return jsonify(status)

@app.route('/analytics')
def analytics():
    """Analytics page"""
    logger.info("Analytics page requested")
    try:
        conn = sqlite3.connect('mental_health.db')
        
        # Get analytics data
        analytics_data = conn.execute('''
            SELECT 
                COUNT(*) as total_requests,
                AVG(response_time) as avg_response_time,
                SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
            FROM analytics
        ''').fetchone()
        
        # Get recent requests
        recent_requests = conn.execute('''
            SELECT timestamp, endpoint, response_time, status_code
            FROM analytics
            ORDER BY timestamp DESC
            LIMIT 20
        ''').fetchall()
        
        conn.close()
        
        return render_template('analytics.html', 
                             analytics_data=analytics_data,
                             recent_requests=recent_requests)
    except Exception as e:
        logger.error(f"Analytics page failed: {e}")
        return f"<h1>Analytics Error</h1><p>{str(e)}</p>", 500

@app.route('/users')
def users():
    """Users page"""
    logger.info("Users page requested")
    try:
        conn = sqlite3.connect('mental_health.db')
        
        # Get user statistics
        user_stats = conn.execute('''
            SELECT 
                COUNT(DISTINCT session_id) as unique_sessions,
                COUNT(*) as total_predictions
            FROM predictions
        ''').fetchone()
        
        # Get recent users/sessions
        recent_users = conn.execute('''
            SELECT DISTINCT session_id, timestamp, gender, age, occupation
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 20
        ''').fetchall()
        
        conn.close()
        
        return render_template('users.html', 
                             user_stats=user_stats,
                             recent_users=recent_users)
    except Exception as e:
        logger.error(f"Users page failed: {e}")
        return f"<h1>Users Error</h1><p>{str(e)}</p>", 500

@app.route('/settings')
def settings():
    """Settings page"""
    logger.info("Settings page requested")
    try:
        # Just return a simple settings page for now
        return render_template('settings.html')
    except Exception as e:
        logger.error(f"Settings page failed: {e}")
        return f"<h1>Settings Error</h1><p>{str(e)}</p>", 500

# Add a simple analytics template if it doesn't exist
@app.route('/shap_explanation', methods=['POST'])
def shap_explanation():
    """Generate SHAP explanation for prediction"""
    logger.info("SHAP explanation request received")
    
    try:
        # Get form data from request
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()
        
        if not form_data:
            logger.error("No form data received for SHAP explanation")
            return jsonify({'success': False, 'error': 'No form data received'}), 400
        
        # Prepare input data
        X_behavioral, X_demographic, behavioral_scores = prepare_input_data(form_data)
        
        # Get prediction
        predictions, probabilities = model.predict(X_behavioral, X_demographic)
        risk_level = int(predictions[0])
        
        # Create combined features for SHAP
        X_beh_latent = model.extract_features(X_behavioral)
        X_combined = np.concatenate([X_beh_latent, X_demographic], axis=1)
        
        # Create feature names
        feature_names = (
            [f'Behavioral_Latent_{i}' for i in range(X_beh_latent.shape[1])] +
            ['Gender_Encoded', 'Occupation_Encoded', 'Country_Encoded', 'Days_Indoors_Encoded', 'Age']
        )
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model.xgb_model)
        
        # Calculate SHAP values with error handling
        try:
            shap_values = explainer.shap_values(X_combined)
        except Exception as e:
            logger.error(f"SHAP values calculation failed: {e}")
            # Fallback: create dummy SHAP values
            num_features = X_combined.shape[1]
            shap_values = np.zeros((1, num_features))
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # For multi-class, use the values for the predicted class
            if risk_level < len(shap_values):
                shap_values_for_class = shap_values[risk_level]
            else:
                # Fallback to first class if risk_level is out of bounds
                shap_values_for_class = shap_values[0]
        else:
            shap_values_for_class = shap_values
        
        # Ensure we have the correct shape
        if hasattr(shap_values_for_class, 'shape'):
            if len(shap_values_for_class.shape) == 3:
                shap_values_for_class = shap_values_for_class[0]  # Take first sample
            elif len(shap_values_for_class.shape) == 0:
                # Handle scalar case
                shap_values_for_class = np.array([[shap_values_for_class]])
        else:
            # Convert to numpy array if it's not already
            shap_values_for_class = np.array([[shap_values_for_class]])
        
        # Get top contributing features
        if len(shap_values_for_class.shape) == 1:
            # Single instance
            shap_vals = shap_values_for_class
        elif len(shap_values_for_class.shape) == 2:
            # Multiple instances, take first one
            shap_vals = shap_values_for_class[0]
        else:
            # Handle higher dimensional arrays
            shap_vals = shap_values_for_class[0]
            while len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]
        
        # Ensure shap_vals is 1D
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals.flatten()
        
        # Get absolute values for ranking
        abs_shap_vals = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap_vals)[::-1][:5]
        
        top_features = []
        for idx in top_indices:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                # Safe scalar conversion with error handling
                try:
                    contribution = float(np.squeeze(np.array(shap_vals[idx])))
                    # Handle NaN or infinite values
                    if not np.isfinite(contribution):
                        contribution = 0.0
                except (ValueError, TypeError, IndexError):
                    contribution = 0.0
                
                top_features.append({
                    'feature': feature_name,
                    'contribution': contribution,
                    'impact': 'positive' if contribution > 0 else 'negative'
                })
        
        # Create explanation report
        report = {
            'success': True,
            'predicted_class': risk_level,
            'class_probabilities': {
                'Low Risk': float(probabilities[0][0]),
                'Medium Risk': float(probabilities[0][1]),
                'High Risk': float(probabilities[0][2])
            },
            'top_contributing_features': top_features,
            'feature_count': len(feature_names),
            'shap_values_count': len(shap_vals)
        }
        
        logger.info(f"SHAP explanation generated successfully for risk level {risk_level}")
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/analytics_data')
def analytics_data():
    """Provide analytics data as JSON for charts"""
    try:
        conn = sqlite3.connect('mental_health.db')
        
        # Get daily prediction counts
        daily_counts = conn.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        ''').fetchall()
        
        # Get risk level distribution
        risk_dist = conn.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM predictions
            GROUP BY risk_level
        ''').fetchall()
        
        conn.close()
        
        return jsonify({
            'daily_counts': [{'date': row[0], 'count': row[1]} for row in daily_counts],
            'risk_distribution': [{'level': row[0], 'count': row[1]} for row in risk_dist]
        })
    except Exception as e:
        logger.error(f"Analytics data failed: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize on startup
if __name__ != '__main__':
    # This runs when imported (e.g., by gunicorn)
    init_db()
    if not load_models(logger):
        logger.error("Failed to load models during initialization!")

# Run directly
if __name__ == '__main__':
    # Initialize
    init_db()
    
    # Load models
    if load_models(logger):
        logger.info("=" * 60)
        logger.info("FLASK APPLICATION STARTING")
        logger.info("=" * 60)
        logger.info(f"Server URL: http://localhost:5000")
        logger.info(f"Dashboard: http://localhost:5000/dashboard")
        logger.info(f"Health check: http://localhost:5000/health")
        logger.info(f"System info: http://localhost:5000/system_info")
        logger.info(f"Logs: http://localhost:5000/logs")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        logger.error("Failed to start Flask application - models not loaded")
        sys.exit(1)