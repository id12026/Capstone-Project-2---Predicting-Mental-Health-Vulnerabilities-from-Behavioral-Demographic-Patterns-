#!/usr/bin/env python3
"""
Diagnostic script for troubleshooting
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import h5py

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import EnhancedLogger

def diagnose_feature_mismatch(app_logger=None):
    """Diagnose feature mismatch issues"""
    if app_logger is None:
        enhanced_logger = EnhancedLogger("diagnose")
        logger = enhanced_logger.get_logger()
    else:
        logger = app_logger
    
    logger.info("=" * 80)
    logger.info("FEATURE MISMATCH DIAGNOSIS")
    logger.info("=" * 80)
    
    issues_found = []
    warnings = []
    ok_items = []
    
    # 1. Check directory structure
    logger.info("\n1. Checking directory structure...")
    required_dirs = ['data', 'models', 'models/preprocessors', 'logs', 'static', 'templates']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            ok_items.append(f"Directory exists: {dir_path}")
        else:
            issues_found.append(f"Missing directory: {dir_path}")
            logger.warning(f"  ❌ Missing: {dir_path}")
    
    # 2. Check data files
    logger.info("\n2. Checking data files...")
    data_files = ['data/mental_health_data.csv', 'data/test_mental_health_data.csv']
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            ok_items.append(f"Data file: {file_path} ({size/1024:.1f} KB)")
            if size == 0:
                issues_found.append(f"Empty file: {file_path}")
        else:
            warnings.append(f"Data file not found: {file_path}")
    
    # 3. Check model files
    logger.info("\n3. Checking model files...")
    model_files = [
        'models/autoencoder_model.h5',
        'models/encoder_model.h5',
        'models/xgb_model.pkl',
        'models/model_info.pkl',
        'models/preprocessors/behavioral_scaler.pkl',
        'models/preprocessors/demographic_scaler.pkl',
        'models/preprocessors/label_encoders.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            ok_items.append(f"Model file: {file_path} ({size/1024:.1f} KB)")
            
            # Check specific files
            if 'scaler' in file_path:
                try:
                    scaler = joblib.load(file_path)
                    if 'behavioral' in file_path:
                        logger.info(f"  ✅ Behavioral scaler expects {scaler.n_features_in_} features")
                    else:
                        logger.info(f"  ✅ Demographic scaler expects {scaler.n_features_in_} features")
                except:
                    issues_found.append(f"Corrupted scaler: {file_path}")
        else:
            issues_found.append(f"Missing model file: {file_path}")
    
    # 4. Check feature dimensions
    logger.info("\n4. Checking feature dimensions...")
    try:
        behavioral_scaler = joblib.load('models/preprocessors/behavioral_scaler.pkl')
        demographic_scaler = joblib.load('models/preprocessors/demographic_scaler.pkl')
        
        logger.info(f"  Behavioral features expected: {behavioral_scaler.n_features_in_}")
        logger.info(f"  Demographic features expected: {demographic_scaler.n_features_in_}")
        logger.info(f"  Total features expected: {behavioral_scaler.n_features_in_ + demographic_scaler.n_features_in_}")
        
        # Check what web app sends
        web_app_behavioral = 7  # Based on our form
        web_app_demographic = 5  # gender, occupation, country, days_indoors, age
        
        if behavioral_scaler.n_features_in_ != web_app_behavioral:
            issues_found.append(f"Behavioral feature mismatch: Model expects {behavioral_scaler.n_features_in_}, Web sends {web_app_behavioral}")
        
        if demographic_scaler.n_features_in_ != web_app_demographic:
            issues_found.append(f"Demographic feature mismatch: Model expects {demographic_scaler.n_features_in_}, Web sends {web_app_demographic}")
        
    except Exception as e:
        issues_found.append(f"Cannot check feature dimensions: {e}")
    
    # 5. Check label encoders
    logger.info("\n5. Checking label encoders...")
    try:
        label_encoders = joblib.load('models/preprocessors/label_encoders.pkl')
        logger.info(f"  Found {len(label_encoders)} label encoders:")
        for feature, encoder in label_encoders.items():
            logger.info(f"    - {feature}: {len(encoder.classes_)} classes")
    except:
        issues_found.append("Label encoders not found or corrupted")
    
    # 6. Check template files
    logger.info("\n6. Checking template files...")
    template_files = ['templates/index.html', 'templates/dashboard.html']
    for file_path in template_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            ok_items.append(f"Template: {file_path} ({size/1024:.1f} KB)")
        else:
            issues_found.append(f"Missing template: {file_path}")
    
    # 7. Check Python environment
    logger.info("\n7. Checking Python environment...")
    try:
        import tensorflow as tf
        logger.info(f"  TensorFlow: {tf.__version__}")
    except:
        issues_found.append("TensorFlow not installed")
    
    try:
        import xgboost as xgb
        logger.info(f"  XGBoost: {xgb.__version__}")
    except:
        issues_found.append("XGBoost not installed")
    
    # 8. Summary
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("=" * 80)
    
    if issues_found:
        logger.error(f"❌ ISSUES FOUND ({len(issues_found)}):")
        for issue in issues_found:
            logger.error(f"  • {issue}")
    else:
        logger.info("✅ No critical issues found!")
    
    if warnings:
        logger.warning(f"⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            logger.warning(f"  • {warning}")
    
    logger.info(f"✓ OK ITEMS ({len(ok_items)}):")
    for ok in ok_items[:10]:  # Show first 10
        logger.info(f"  • {ok}")
    
    if len(ok_items) > 10:
        logger.info(f"  ... and {len(ok_items) - 10} more")
    
    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)
    
    if issues_found:
        logger.info("1. Run quick fix: python quick_fix.py")
        logger.info("2. Train model: python main.py --mode train")
        logger.info("3. Test system: python main.py --mode test")
    else:
        logger.info("1. Start web server: python main.py --mode serve")
        logger.info("2. Test prediction: python main.py --mode test")
    
    logger.info("=" * 80)
    
    return len(issues_found) == 0

if __name__ == '__main__':
    diagnose_feature_mismatch()