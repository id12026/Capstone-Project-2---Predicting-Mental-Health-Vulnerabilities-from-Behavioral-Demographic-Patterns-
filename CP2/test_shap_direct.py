#!/usr/bin/env python3
"""
Direct test of SHAP functionality without HTTP
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from app import prepare_input_data, load_models
from utils.logger import EnhancedLogger

# Initialize logger
logger = EnhancedLogger("test_shap").get_logger()

def test_shap_directly():
    """Test SHAP functionality directly"""
    
    # Sample form data
    test_data = {
        'gender': 'Male',
        'age': '25',
        'occupation': 'Student',
        'country': 'United States',
        'family_history': 'No',
        'days_indoors': '1-14 days',
        'growing_stress': 'Medium',
        'mood_swings': 'Low',
        'coping_struggles': 'Sometimes',
        'work_interest': 'No',
        'social_weakness': 'No',
        'changes_habits': 'No'
    }
    
    try:
        print("Loading models...")
        if not load_models(logger):
            print("❌ Failed to load models")
            return False
        
        from app import model
        
        print("Preparing input data...")
        X_behavioral, X_demographic, behavioral_scores = prepare_input_data(test_data)
        
        print("Getting prediction...")
        predictions, probabilities = model.predict(X_behavioral, X_demographic)
        risk_level = int(predictions[0])
        
        print("Creating combined features for SHAP...")
        X_beh_latent = model.extract_features(X_behavioral)
        X_combined = np.concatenate([X_beh_latent, X_demographic], axis=1)
        
        print("Testing SHAP calculation...")
        import shap
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model.xgb_model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_combined)
        
        print("✅ SHAP values calculated successfully!")
        print(f"SHAP values type: {type(shap_values)}")
        print(f"SHAP values shape: {shap_values[0].shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # Test the conversion logic
        if isinstance(shap_values, list):
            shap_values_for_class = shap_values[risk_level]
        else:
            shap_values_for_class = shap_values
        
        # Test the safe conversion
        if len(shap_values_for_class.shape) == 1:
            shap_vals = shap_values_for_class
        else:
            shap_vals = shap_values_for_class[0]
        
        # Test scalar conversion
        for i in range(min(5, len(shap_vals))):
            try:
                contribution = float(np.squeeze(np.array(shap_vals[i])))
                if not np.isfinite(contribution):
                    contribution = 0.0
                print(f"Feature {i}: {contribution}")
            except Exception as e:
                print(f"Error converting feature {i}: {e}")
        
        print("✅ SHAP functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during SHAP test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_shap_directly()
