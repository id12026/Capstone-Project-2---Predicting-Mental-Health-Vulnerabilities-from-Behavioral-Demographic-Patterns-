#!/usr/bin/env python3
"""
Quick fix for common issues with logging
"""

import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import h5py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import EnhancedLogger

def quick_fix(app_logger=None):
    """Apply quick fixes for common issues"""
    if app_logger is None:
        enhanced_logger = EnhancedLogger("quick_fix")
        logger = enhanced_logger.get_logger()
    else:
        logger = app_logger
    
    logger.info("=" * 60)
    logger.info("APPLYING QUICK FIXES")
    logger.info("=" * 60)
    
    fixes_applied = []
    
    try:
        # 1. Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            os.makedirs('models/preprocessors', exist_ok=True)
            fixes_applied.append("Created models directory")
            logger.info("✅ Created models directory")
        
        # 2. Create dummy scalers with correct dimensions
        # Behavioral scaler: 7 features
        behavioral_scaler = StandardScaler()
        dummy_behavioral = np.random.randn(100, 7)
        behavioral_scaler.fit(dummy_behavioral)
        joblib.dump(behavioral_scaler, 'models/preprocessors/behavioral_scaler.pkl')
        fixes_applied.append("Created behavioral scaler (7 features)")
        logger.info(f"✅ Behavioral scaler created: expects {behavioral_scaler.n_features_in_} features")
        
        # Demographic scaler: 5 features
        demographic_scaler = StandardScaler()
        dummy_demographic = np.random.randn(100, 5)
        demographic_scaler.fit(dummy_demographic)
        joblib.dump(demographic_scaler, 'models/preprocessors/demographic_scaler.pkl')
        fixes_applied.append("Created demographic scaler (5 features)")
        logger.info(f"✅ Demographic scaler created: expects {demographic_scaler.n_features_in_} features")
        
        # 3. Create label encoders
        label_encoders = {}
        
        # Gender encoder
        le_gender = LabelEncoder()
        le_gender.fit(['Male', 'Female', 'Other', 'Unknown', 'Prefer not to say'])
        label_encoders['gender'] = le_gender
        
        # Occupation encoder
        le_occupation = LabelEncoder()
        le_occupation.fit(['Student', 'Corporate', 'Healthcare', 'IT', 'Self_Employed', 
                          'Unemployed', 'Retired', 'Other', 'Unknown'])
        label_encoders['occupation'] = le_occupation
        
        # Country encoder
        le_country = LabelEncoder()
        le_country.fit(['United States', 'India', 'United Kingdom', 'Canada', 'Australia',
                       'Germany', 'France', 'Japan', 'Other', 'Unknown'])
        label_encoders['country'] = le_country
        
        # Days indoors encoder
        le_days = LabelEncoder()
        le_days.fit(['1-14 days', '15-30 days', '31-60 days', 'More than 2 months', 'Unknown'])
        label_encoders['days_indoors'] = le_days
        
        joblib.dump(label_encoders, 'models/preprocessors/label_encoders.pkl')
        fixes_applied.append("Created label encoders")
        logger.info(f"✅ Label encoders created: {len(label_encoders)} features")
        
        # 4. Create dummy models
        import xgboost as xgb
        
        # XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False
        )
        
        # Train on dummy data
        dummy_X = np.random.randn(100, 12)  # 7 behavioral + 5 demographic
        dummy_y = np.random.choice([0, 1, 2], 100, p=[0.7, 0.2, 0.1])
        xgb_model.fit(dummy_X, dummy_y)
        
        joblib.dump(xgb_model, 'models/xgb_model.pkl')
        fixes_applied.append("Created XGBoost model")
        logger.info("✅ XGBoost model created")
        
        # 5. Create dummy autoencoder files
        with h5py.File('models/autoencoder_model.h5', 'w') as f:
            f.attrs['model_type'] = 'dummy_autoencoder'
            f.attrs['input_dim'] = 7
            f.attrs['latent_dim'] = 3
        
        with h5py.File('models/encoder_model.h5', 'w') as f:
            f.attrs['model_type'] = 'dummy_encoder'
            f.attrs['input_dim'] = 7
            f.attrs['latent_dim'] = 3
        
        fixes_applied.append("Created dummy autoencoder files")
        logger.info("✅ Autoencoder files created")
        
        # 6. Create model info
        model_info = {
            'input_dim_behavioral': 7,
            'latent_dim': 3,
            'input_dim_demographic': 5,
            'timestamp': np.datetime64('now'),
            'is_dummy': True,
            'version': '1.0.0'
        }
        joblib.dump(model_info, 'models/model_info.pkl')
        fixes_applied.append("Created model info file")
        
        # 7. Create sample data if none exists
        if not os.path.exists('data/mental_health_data.csv'):
            from create_test_data import create_test_dataset
            create_test_dataset()
            fixes_applied.append("Created sample dataset")
            logger.info("✅ Sample dataset created")
        
        logger.info("=" * 60)
        logger.info("QUICK FIXES APPLIED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Applied fixes:")
        for fix in fixes_applied:
            logger.info(f"  ✓ {fix}")
        logger.info("=" * 60)
        logger.info("You can now run: python main.py --mode serve")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Quick fixes failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    quick_fix()