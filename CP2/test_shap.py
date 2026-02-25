#!/usr/bin/env python3
"""
Test script for SHAP explanation functionality
"""
import requests
import json

def test_shap_explanation():
    """Test the SHAP explanation endpoint"""
    
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
        # Test the SHAP explanation endpoint
        print("Testing SHAP explanation endpoint...")
        response = requests.post(
            'http://localhost:5000/shap_explanation',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ SHAP explanation generated successfully!")
                print(f"Predicted Class: {result.get('predicted_class')}")
                print(f"Top Features: {result.get('top_contributing_features', [])}")
                return True
            else:
                print(f"❌ SHAP explanation failed: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_shap_explanation()
