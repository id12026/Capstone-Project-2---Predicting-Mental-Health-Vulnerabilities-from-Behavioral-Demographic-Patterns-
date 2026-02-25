
# deployment.py
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPredictor:
    """Deployment-ready mental health vulnerability predictor"""
    
    def __init__(self, model_path='saved_models/best_model.pkl',
                 scaler_path='saved_models/scaler.pkl',
                 encoders_path='saved_models/label_encoders.pkl',
                 features_path='saved_models/selected_features.pkl'):
        
        # Load all components
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        self.selected_features = joblib.load(features_path)
        
        # Define mappings
        self.days_mapping = {
            '1-14 days': 7,
            '15-30 days': 22,
            '31-60 days': 45,
            'More than 2 months': 90
        }
        
        self.binary_mapping = {'Yes': 1, 'No': 0, 'Not sure': 0.5}
        self.mood_mapping = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
        
    def preprocess_input(self, input_dict):
        """Preprocess user input similar to training pipeline"""
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Apply feature engineering
        # Map Days_Indoors
        if 'Days_Indoors' in input_df.columns:
            input_df['Days_Indoors_numeric'] = input_df['Days_Indoors'].map(
                self.days_mapping).fillna(0)
        
        # Map binary features
        binary_cols = ['Social_Weakness', 'Growing_Stress', 'Coping_Struggles',
                      'care_options', 'mental_health_interview', 'Work_Interest', 'Changes_Habits']
        
        for col in binary_cols:
            if col in input_df.columns:
                input_df[f'{col}_numeric'] = input_df[col].map(
                    self.binary_mapping).fillna(0)
        
        # Map Mood_Swings
        if 'Mood_Swings' in input_df.columns:
            input_df['Mood_Swings_numeric'] = input_df['Mood_Swings'].map(
                self.mood_mapping).fillna(0.5)
        
        # Encode categorical variables
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels
                val = input_df[col].iloc[0]
                if val not in le.classes_:
                    # Use most common class for unseen labels
                    input_df[col] = le.classes_[0]
                else:
                    input_df[col] = le.transform(input_df[col])
        
        # Select and align features
        input_aligned = input_df.reindex(columns=self.selected_features, fill_value=0)
        
        return input_aligned
    
    def predict(self, input_dict):
        """Make prediction for single input"""
        
        # Preprocess
        processed_input = self.preprocess_input(input_dict)
        
        # Scale
        scaled_input = self.scaler.transform(processed_input)
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(scaled_input)[0, 1]
        else:
            probability = float(self.model.predict(scaled_input)[0])
        
        # Calculate score
        vulnerability_score = probability * 100
        
        # Determine risk level
        if vulnerability_score < 30:
            risk_level = 'Low Risk'
            risk_color = 'green'
        elif vulnerability_score < 70:
            risk_level = 'Medium Risk'
            risk_color = 'orange'
        else:
            risk_level = 'High Risk'
            risk_color = 'red'
        
        return {
            'vulnerability_score': round(vulnerability_score, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probability': round(probability, 3)
        }
    
    def batch_predict(self, input_list):
        """Make predictions for multiple inputs"""
        return [self.predict(inp) for inp in input_list]

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = MentalHealthPredictor()
    
    # Example input
    example_input = {
        'Gender': 'Male',
        'Country': 'United States',
        'Occupation': 'Corporate',
        'self_employed': 'No',
        'family_history': 'Yes',
        'Days_Indoors': '15-30 days',
        'Growing_Stress': 'Yes',
        'Changes_Habits': 'Yes',
        'Mental_Health_History': 'Yes',
        'Mood_Swings': 'High',
        'Coping_Struggles': 'Yes',
        'Work_Interest': 'No',
        'Social_Weakness': 'Yes',
        'mental_health_interview': 'No',
        'care_options': 'Not sure'
    }
    
    # Make prediction
    result = predictor.predict(example_input)
    print("Prediction Result:")
    print(f"Vulnerability Score: {result['vulnerability_score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probability: {result['probability']}")
