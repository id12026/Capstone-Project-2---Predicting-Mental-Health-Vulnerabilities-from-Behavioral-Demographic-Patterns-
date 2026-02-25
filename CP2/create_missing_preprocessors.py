from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create missing preprocessors
os.makedirs('models/preprocessors', exist_ok=True)

# Create behavioral scaler
behavioral_scaler = StandardScaler()
joblib.dump(behavioral_scaler, 'models/preprocessors/behavioral_scaler.pkl')
print('Created behavioral_scaler.pkl')

# Create demographic scaler
demographic_scaler = StandardScaler()
joblib.dump(demographic_scaler, 'models/preprocessors/demographic_scaler.pkl')
print('Created demographic_scaler.pkl')

print('All missing preprocessors created successfully!')
