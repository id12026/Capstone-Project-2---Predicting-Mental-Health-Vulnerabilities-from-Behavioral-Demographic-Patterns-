import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, logger=None):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.logger = logger
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess(self, filepath):
        """Load and preprocess the dataset"""
        self.logger.info(f"Loading dataset from: {filepath}")
        
        try:
            # Try different delimiters
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(filepath, sep=sep, engine='python', on_bad_lines='warn')
                    if len(df.columns) > 1:
                        self.logger.info(f"Successfully loaded with separator '{sep}': {df.shape}")
                        break
                except:
                    continue
            else:
                # If all separators fail, try reading raw and parsing
                self.logger.warning("Standard CSV reading failed, trying manual parsing...")
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 0:
                    # Try to detect separator from first line
                    first_line = lines[0].strip()
                    if '\t' in first_line:
                        sep = '\t'
                    elif ',' in first_line:
                        sep = ','
                    else:
                        sep = ' '
                    
                    headers = first_line.split(sep)
                    data = []
                    for line in lines[1:]:
                        if line.strip():
                            data.append(line.strip().split(sep))
                    
                    df = pd.DataFrame(data, columns=headers)
                    self.logger.info(f"Manually parsed dataset: {df.shape}")
                else:
                    raise ValueError("File is empty")
            
            self.logger.info(f"Dataset shape: {df.shape}")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            
            # Clean column names
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            self.logger.info(f"Cleaned columns: {df.columns.tolist()}")
            
            # Show data types
            self.logger.info(f"Data types:\n{df.dtypes}")
            
            # Show sample
            self.logger.info(f"First 3 rows:\n{df.head(3).to_string()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def clean_and_transform_data(self, df):
        """Clean and transform the raw data"""
        self.logger.info("Cleaning and transforming data...")
        
        df_clean = df.copy()
        
        # Clean each column
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Replace empty strings with NaN
                df_clean[col] = df_clean[col].replace(['', 'nan', 'NaN', 'None'], np.nan)
        
        # Report missing values
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            self.logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        
        self.logger.info(f"Data shape after cleaning: {df_clean.shape}")
        return df_clean
    
    def create_target_variable(self, df):
        """Create target variable from existing features"""
        self.logger.info("Creating target variable...")
        
        df_target = df.copy()
        
        # List of possible behavioral columns
        behavioral_columns = [
            'Growing_Stress', 'Mood_Swings', 'Coping_Struggles',
            'Work_Interest', 'Social_Weakness', 'Changes_Habits',
            'treatment', 'family_history', 'Mental_Health_History'
        ]
        
        # Find which columns actually exist
        existing_cols = [col for col in behavioral_columns if col in df_target.columns]
        self.logger.info(f"Found behavioral columns: {existing_cols}")
        
        # Initialize behavioral score
        df_target['behavioral_score'] = 0
        
        # Define mappings
        stress_map = {'Yes': 2, 'yes': 2, 'Maybe': 1, 'maybe': 1, 'No': 0, 'no': 0, 'Low': 0, 'Medium': 1, 'High': 2}
        binary_map = {'Yes': 1, 'yes': 1, 'No': 0, 'no': 0}
        
        # Calculate scores for existing columns
        for col in existing_cols:
            if col in ['Growing_Stress', 'Mood_Swings']:
                df_target[f'{col}_score'] = df_target[col].map(stress_map).fillna(0)
            else:
                df_target[f'{col}_score'] = df_target[col].map(binary_map).fillna(0)
            
            df_target['behavioral_score'] += df_target[f'{col}_score']
            self.logger.info(f"  Added {col}: score range {df_target[f'{col}_score'].min()}-{df_target[f'{col}_score'].max()}")
        
        self.logger.info(f"Behavioral score range: {df_target['behavioral_score'].min()} to {df_target['behavioral_score'].max()}")
        
        # Create risk levels
        max_score = df_target['behavioral_score'].max()
        if max_score <= 3:
            bins = [-1, 1, 2, 3]
        elif max_score <= 6:
            bins = [-1, 2, 4, 6]
        elif max_score <= 10:
            bins = [-1, 3, 6, 10]
        else:
            bins = [-1, 4, 8, max_score]
        
        df_target['risk_level'] = pd.cut(df_target['behavioral_score'], 
                                      bins=bins, 
                                      labels=[0, 1, 2])
        
        # Log distribution
        distribution = df_target['risk_level'].value_counts()
        percentages = df_target['risk_level'].value_counts(normalize=True) * 100
        
        self.logger.info("Target distribution:")
        for level in [0, 1, 2]:
            count = distribution.get(level, 0)
            percent = percentages.get(level, 0)
            self.logger.info(f"  Level {level}: {count} samples ({percent:.1f}%)")
        
        return df_target
    
    def encode_features(self, df):
        """Encode categorical features"""
        self.logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        # Common categorical columns
        categorical_cols = [
            'Gender', 'Country', 'Occupation', 'self_employed',
            'family_history', 'Days_Indoors', 'Mental_Health_History'
        ]
        
        # Only encode columns that exist
        existing_cols = [col for col in categorical_cols if col in df_encoded.columns]
        
        for col in existing_cols:
            try:
                # Fill missing values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                
                # Create and fit encoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                
                self.logger.info(f"  Encoded {col}: {len(le.classes_)} classes")
            except Exception as e:
                self.logger.warning(f"  Could not encode {col}: {e}")
        
        return df_encoded
    
    def prepare_features(self, df):
        """Prepare feature sets for modeling with train/val/test split"""
        self.logger.info("Preparing features for modeling with train/val/test split...")
        
        # Behavioral features (scores)
        behavioral_score_cols = [col for col in df.columns if col.endswith('_score')]
        self.logger.info(f"Behavioral score columns: {behavioral_score_cols}")
        
        # Demographic features
        demographic_cols = [
            'Gender', 'Age', 'Country', 'Occupation',
            'self_employed', 'Days_Indoors'
        ]
        
        # Only use existing columns
        existing_demo = [col for col in demographic_cols if col in df.columns]
        
        # If Age column doesn't exist, try to create it
        if 'Age' not in df.columns and 'age' in df.columns.str.lower():
            age_col = [col for col in df.columns if 'age' in col.lower()][0]
            df = df.rename(columns={age_col: 'Age'})
            existing_demo.append('Age')
        
        self.logger.info(f"Demographic columns: {existing_demo}")
        
        # Check if we have features
        if not behavioral_score_cols and not existing_demo:
            self.logger.error("No features found for modeling!")
            return None
        
        # Create feature matrices
        X_behavioral = df[behavioral_score_cols].values if behavioral_score_cols else np.zeros((len(df), 1))
        X_demographic = df[existing_demo].fillna(0).values if existing_demo else np.zeros((len(df), 1))
        y = df['risk_level'].values
        
        self.logger.info(f"Feature shapes: X_behavioral={X_behavioral.shape}, X_demographic={X_demographic.shape}, y={y.shape}")
        
        # First split: separate test set (20%)
        X_beh_temp, X_beh_test, X_demo_temp, X_demo_test, y_temp, y_test = train_test_split(
            X_behavioral, X_demographic, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        # Second split: divide remaining 80% into train (60%) and validation (20%)
        # This gives us 60%/20%/20% split
        X_beh_train, X_beh_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
            X_beh_temp, X_demo_temp, y_temp,
            test_size=0.25,  # 0.25 of 0.8 = 0.2 of total
            random_state=42,
            stratify=y_temp
        )
        
        self.logger.info("Train/Validation/Test split completed:")
        self.logger.info(f"  Train: X_beh={X_beh_train.shape}, X_demo={X_demo_train.shape}, y={y_train.shape}")
        self.logger.info(f"  Validation: X_beh={X_beh_val.shape}, X_demo={X_demo_val.shape}, y={y_val.shape}")
        self.logger.info(f"  Test:  X_beh={X_beh_test.shape}, X_demo={X_demo_test.shape}, y={y_test.shape}")
        
        # Scale features using training data only
        if X_beh_train.shape[1] > 0:
            X_beh_train_scaled = self.scaler.fit_transform(X_beh_train)
            X_beh_val_scaled = self.scaler.transform(X_beh_val)
            X_beh_test_scaled = self.scaler.transform(X_beh_test)
        else:
            X_beh_train_scaled = X_beh_train
            X_beh_val_scaled = X_beh_val
            X_beh_test_scaled = X_beh_test
        
        # Use separate scaler for demographic features
        demo_scaler = StandardScaler()
        if X_demo_train.shape[1] > 0:
            X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
            X_demo_val_scaled = demo_scaler.transform(X_demo_val)
            X_demo_test_scaled = demo_scaler.transform(X_demo_test)
        else:
            X_demo_train_scaled = X_demo_train
            X_demo_val_scaled = X_demo_val
            X_demo_test_scaled = X_demo_test
        
        # Save the demographic scaler
        self.demo_scaler = demo_scaler
        
        return {
            'X_beh_train': X_beh_train_scaled,
            'X_beh_val': X_beh_val_scaled,
            'X_beh_test': X_beh_test_scaled,
            'X_demo_train': X_demo_train_scaled,
            'X_demo_val': X_demo_val_scaled,
            'X_demo_test': X_demo_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'behavioral_feature_names': behavioral_score_cols,
            'demographic_feature_names': existing_demo,
            'df': df
        }
    
    def save_encoders(self, path='models/preprocessors/'):
        """Save encoders and scalers"""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.label_encoders, f'{path}label_encoders.pkl')
        joblib.dump(self.scaler, f'{path}behavioral_scaler.pkl')
        joblib.dump(self.demo_scaler, f'{path}demographic_scaler.pkl')
        
        self.logger.info(f"Preprocessors saved to {path}")
    
    def load_encoders(self, path='models/preprocessors/'):
        """Load encoders and scalers"""
        self.label_encoders = joblib.load(f'{path}label_encoders.pkl')
        self.scaler = joblib.load(f'{path}behavioral_scaler.pkl')
        self.demo_scaler = joblib.load(f'{path}demographic_scaler.pkl')
        
        self.logger.info(f"Preprocessors loaded from {path}")