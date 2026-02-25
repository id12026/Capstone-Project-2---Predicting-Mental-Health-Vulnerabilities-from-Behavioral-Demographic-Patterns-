import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import traceback

class MentalHealthModel:
    def __init__(self, input_dim_behavioral=6, latent_dim=3, input_dim_demographic=8, logger=None):
        self.input_dim_behavioral = input_dim_behavioral
        self.latent_dim = latent_dim
        self.input_dim_demographic = input_dim_demographic
        self.autoencoder = None
        self.encoder = None
        self.xgb_model = None
        self.logger = logger
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
    
    def build_autoencoder(self):
        """Build and compile autoencoder for behavioral feature extraction with regularization"""
        self.logger.info("Building autoencoder model with regularization...")
        
        try:
            # Encoder with moderate regularization and increased capacity
            encoder_input = keras.Input(shape=(self.input_dim_behavioral,))
            x = layers.Dense(48, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='enc_dense1')(encoder_input)
            x = layers.Dropout(0.3)(x)  # Moderate dropout
            x = layers.Dense(24, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='enc_dense2')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(12, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='enc_dense3')(x)
            latent = layers.Dense(self.latent_dim, activation='relu', 
                                kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                                name='latent')(x)
            
            # Decoder with moderate regularization and increased capacity
            x = layers.Dense(12, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='dec_dense1')(latent)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(24, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='dec_dense2')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(48, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                           name='dec_dense3')(x)
            decoder_output = layers.Dense(self.input_dim_behavioral, activation='sigmoid', 
                                        kernel_regularizer=keras.regularizers.l2(0.005),  # Moderate L2 regularization
                                        name='output')(x)
            
            # Autoencoder model
            self.autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
            self.encoder = keras.Model(encoder_input, latent, name='encoder')
            
            # Compile with moderate learning rate for balanced training
            self.autoencoder.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Moderate learning rate
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info("Autoencoder architecture built with L2 regularization:")
            self.autoencoder.summary(print_fn=self.logger.info)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build autoencoder: {e}")
            return False
    
    def train_autoencoder(self, X_train, X_val, X_test, epochs=100, batch_size=32):
        """Train the autoencoder with train/validation/test split and enhanced regularization"""
        self.logger.info(f"Training autoencoder for {epochs} epochs with batch size {batch_size}...")
        
        try:
            # Moderate early stopping with validation monitoring
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=12,  # Moderate patience
                restore_best_weights=True,
                min_delta=0.001  # Lower minimum improvement threshold
            )
            
            # Moderate learning rate scheduler
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Moderate reduction
                patience=8,  # Moderate patience
                min_lr=1e-7,  # Reasonable minimum learning rate
                verbose=1
            )
            
            # Add model checkpoint to save best model
            checkpoint = keras.callbacks.ModelCheckpoint(
                'models/best_autoencoder.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            
            # Add tensorboard callback for monitoring (optional)
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='logs/autoencoder_tensorboard',
                histogram_freq=1,
                write_graph=True
            )
            
            # Train the model with validation data
            history = self.autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, X_val),
                callbacks=[early_stopping, lr_scheduler, checkpoint],
                verbose=1  # Show training progress
            )
            
            # Log training progress
            self.logger.info("Autoencoder training completed:")
            self.logger.info(f"  Final training loss: {history.history['loss'][-1]:.4f}")
            self.logger.info(f"  Final validation loss: {history.history['val_loss'][-1]:.4f}")
            self.logger.info(f"  Final training MAE: {history.history['mae'][-1]:.4f}")
            self.logger.info(f"  Final validation MAE: {history.history['val_mae'][-1]:.4f}")
            self.logger.info(f"  Trained for {len(history.history['loss'])} epochs")
            
            # Evaluate on test set
            test_loss, test_mae = self.autoencoder.evaluate(X_test, X_test, verbose=0)
            self.logger.info(f"  Test loss: {test_loss:.4f}")
            self.logger.info(f"  Test MAE: {test_mae:.4f}")
            
            # Plot training history
            self.plot_training_history(history)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to train autoencoder: {e}")
            return None
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            axes[0].plot(history.history['loss'], label='Training Loss')
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0].set_title('Autoencoder Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot MAE
            axes[1].plot(history.history['mae'], label='Training MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('Autoencoder MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('static', exist_ok=True)
            plot_path = 'static/autoencoder_training.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save training plot: {e}")
    
    def extract_features(self, X_behavioral):
        """Extract latent features using encoder"""
        self.logger.info(f"Extracting latent features from {X_behavioral.shape[0]} samples...")
        
        try:
            features = self.encoder.predict(X_behavioral, verbose=0)
            self.logger.info(f"Extracted features shape: {features.shape}")
            return features
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return X_behavioral  # Fallback to original features
    
    def build_hybrid_model(self, X_beh_train_latent, X_demo_train, y_train, 
                          X_beh_val_latent=None, X_demo_val=None, y_val=None):
        """Build and train XGBoost on combined features with validation and regularization"""
        self.logger.info("Building hybrid model (Autoencoder + XGBoost) with validation...")
        
        try:
            # Combine training features
            X_train_combined = np.concatenate([X_beh_train_latent, X_demo_train], axis=1)
            self.logger.info(f"Training features shape: {X_train_combined.shape}")
            
            # Combine validation features if provided
            eval_set = None
            if X_beh_val_latent is not None and X_demo_val is not None and y_val is not None:
                X_val_combined = np.concatenate([X_beh_val_latent, X_demo_val], axis=1)
                eval_set = [(X_train_combined, y_train), (X_val_combined, y_val)]
                self.logger.info(f"Validation features shape: {X_val_combined.shape}")
            
            # Train XGBoost with moderate regularization for target accuracy
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,  # Moderate number of estimators
                max_depth=5,  # Moderate depth for better learning
                learning_rate=0.02,  # Moderate learning rate
                subsample=0.8,  # Higher subsample for more data utilization
                colsample_bytree=0.8,  # Higher feature sampling
                reg_alpha=0.05,  # Lower L1 regularization
                reg_lambda=0.5,  # Lower L2 regularization
                min_child_weight=2,  # Lower minimum sum of instance weight
                gamma=0.1,  # Lower minimum loss reduction
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss',
                early_stopping_rounds=15  # Moderate early stopping
            )
            
            self.logger.info("Training XGBoost model with validation...")
            self.xgb_model.fit(
                X_train_combined, y_train,
                eval_set=eval_set,
                verbose=True
            )
            
            # Feature importance
            importance = self.xgb_model.feature_importances_
            self.logger.info(f"XGBoost model trained successfully!")
            self.logger.info(f"Feature importance range: {importance.min():.4f} - {importance.max():.4f}")
            
            # Log best iteration if early stopping was used
            if hasattr(self.xgb_model, 'best_iteration'):
                self.logger.info(f"Best iteration: {self.xgb_model.best_iteration}")
                self.logger.info(f"Best score: {self.xgb_model.best_score:.4f}")
            
            return X_train_combined
            
        except Exception as e:
            self.logger.error(f"Failed to build hybrid model: {e}")
            return None
    
    def predict(self, X_behavioral, X_demographic):
        """Make predictions using the hybrid model"""
        try:
            # Extract latent features
            X_behavioral_latent = self.extract_features(X_behavioral)
            
            # Combine features
            X_combined = np.concatenate([X_behavioral_latent, X_demographic], axis=1)
            
            # Make predictions
            predictions = self.xgb_model.predict(X_combined)
            probabilities = self.xgb_model.predict_proba(X_combined)
            
            return predictions, probabilities
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return dummy predictions
            n_samples = X_behavioral.shape[0]
            return np.zeros(n_samples), np.zeros((n_samples, 3))
    
    def evaluate_model(self, X_behavioral_test, X_demographic_test, y_test, 
                      X_behavioral_val=None, X_demographic_val=None, y_val=None):
        """Evaluate model performance on test and validation sets"""
        self.logger.info("Evaluating model performance on test and validation sets...")
        
        results = {}
        
        try:
            # Test set evaluation
            self.logger.info("\n--- TEST SET EVALUATION ---")
            test_predictions, test_probabilities = self.predict(X_behavioral_test, X_demographic_test)
            
            # Calculate test metrics
            test_accuracy = accuracy_score(y_test, test_predictions)
            test_precision = precision_score(y_test, test_predictions, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, test_predictions, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, test_predictions, average='weighted', zero_division=0)
            
            self.logger.info("=" * 60)
            self.logger.info("TEST SET RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"Accuracy:  {test_accuracy:.4f}")
            self.logger.info(f"Precision: {test_precision:.4f}")
            self.logger.info(f"Recall:    {test_recall:.4f}")
            self.logger.info(f"F1-Score:  {test_f1:.4f}")
            self.logger.info("=" * 60)
            
            # Test classification report
            test_report = classification_report(y_test, test_predictions, 
                                              target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                                              zero_division=0)
            self.logger.info("Test Classification Report:\n" + test_report)
            
            # Validation set evaluation if provided
            if X_behavioral_val is not None and X_demographic_val is not None and y_val is not None:
                self.logger.info("\n--- VALIDATION SET EVALUATION ---")
                val_predictions, val_probabilities = self.predict(X_behavioral_val, X_demographic_val)
                
                # Calculate validation metrics
                val_accuracy = accuracy_score(y_val, val_predictions)
                val_precision = precision_score(y_val, val_predictions, average='weighted', zero_division=0)
                val_recall = recall_score(y_val, val_predictions, average='weighted', zero_division=0)
                val_f1 = f1_score(y_val, val_predictions, average='weighted', zero_division=0)
                
                self.logger.info("=" * 60)
                self.logger.info("VALIDATION SET RESULTS")
                self.logger.info("=" * 60)
                self.logger.info(f"Accuracy:  {val_accuracy:.4f}")
                self.logger.info(f"Precision: {val_precision:.4f}")
                self.logger.info(f"Recall:    {val_recall:.4f}")
                self.logger.info(f"F1-Score:  {val_f1:.4f}")
                self.logger.info("=" * 60)
                
                # Validation classification report
                val_report = classification_report(y_val, val_predictions, 
                                                 target_names=['Low Risk', 'Medium Risk', 'High Risk'],
                                                 zero_division=0)
                self.logger.info("Validation Classification Report:\n" + val_report)
                
                # Check for overfitting
                accuracy_diff = abs(test_accuracy - val_accuracy)
                if accuracy_diff > 0.05:  # 5% difference threshold
                    self.logger.warning(f"⚠️  Potential overfitting detected! Accuracy difference: {accuracy_diff:.4f}")
                    self.logger.warning(f"  Validation: {val_accuracy:.4f} vs Test: {test_accuracy:.4f}")
                else:
                    self.logger.info(f"✅ No significant overfitting detected (difference: {accuracy_diff:.4f})")
                
                results['val_accuracy'] = val_accuracy
                results['val_precision'] = val_precision
                results['val_recall'] = val_recall
                results['val_f1'] = val_f1
                results['val_predictions'] = val_predictions
                results['val_probabilities'] = val_probabilities
            
            # Confusion matrix for test set
            self.plot_confusion_matrix(y_test, test_predictions)
            
            results.update({
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_predictions': test_predictions,
                'test_probabilities': test_probabilities,
                'accuracy_diff': accuracy_diff if 'accuracy_diff' in locals() else 0
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Low', 'Medium', 'High'],
                       yticklabels=['Low', 'Medium', 'High'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('static', exist_ok=True)
            plot_path = 'static/confusion_matrix.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save confusion matrix: {e}")
    
    def save_models(self, path='models/'):
        """Save trained models"""
        self.logger.info(f"Saving models to {path}...")
        
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save autoencoder
            autoencoder_path = f'{path}autoencoder_model.h5'
            encoder_path = f'{path}encoder_model.h5'
            
            # Save with custom_objects to preserve configuration
            self.autoencoder.save(autoencoder_path, save_format='h5', include_optimizer=True)
            self.encoder.save(encoder_path, save_format='h5', include_optimizer=True)
            
            # Save XGBoost model
            xgb_path = f'{path}xgb_model.pkl'
            joblib.dump(self.xgb_model, xgb_path)
            
            self.logger.info(f"✅ Models saved successfully:")
            self.logger.info(f"  Autoencoder: {autoencoder_path}")
            self.logger.info(f"  Encoder: {encoder_path}")
            self.logger.info(f"  XGBoost: {xgb_path}")
            
            # Save model info
            model_info = {
                'input_dim_behavioral': self.input_dim_behavioral,
                'latent_dim': self.latent_dim,
                'input_dim_demographic': self.input_dim_demographic,
                'timestamp': np.datetime64('now')
            }
            joblib.dump(model_info, f'{path}model_info.pkl')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, path='models/'):
        """Load trained models"""
        self.logger.info(f"Loading models from {path}...")
        
        try:
            # Load autoencoder
            autoencoder_path = f'{path}autoencoder_model.h5'
            encoder_path = f'{path}encoder_model.h5'
            
            if not os.path.exists(autoencoder_path):
                self.logger.error(f"Autoencoder not found at {autoencoder_path}")
                return False
            
            # Load models with custom_objects to handle potential issues
            try:
                self.autoencoder = keras.models.load_model(autoencoder_path)
            except Exception as ae:
                self.logger.warning(f"Failed to load autoencoder directly: {ae}")
                # Try alternative loading approach
                self.autoencoder = keras.models.load_model(autoencoder_path, compile=False)
            
            try:
                self.encoder = keras.models.load_model(encoder_path)
            except Exception as ee:
                self.logger.warning(f"Failed to load encoder directly: {ee}")
                # Try alternative loading approach
                self.encoder = keras.models.load_model(encoder_path, compile=False)
            
            # Load XGBoost model
            xgb_path = f'{path}xgb_model.pkl'
            if not os.path.exists(xgb_path):
                self.logger.error(f"XGBoost model not found at {xgb_path}")
                return False
            
            self.xgb_model = joblib.load(xgb_path)
            
            # Load model info
            info_path = f'{path}model_info.pkl'
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                self.input_dim_behavioral = model_info.get('input_dim_behavioral', 6)
                self.latent_dim = model_info.get('latent_dim', 3)
                self.input_dim_demographic = model_info.get('input_dim_demographic', 8)
            
            self.logger.info("✅ Models loaded successfully!")
            self.logger.info(f"  Autoencoder input dim: {self.input_dim_behavioral}")
            self.logger.info(f"  Latent dim: {self.latent_dim}")
            self.logger.info(f"  Demographic dim: {self.input_dim_demographic}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False


def train_complete_model(data_file, epochs=50, batch_size=32, logger=None):
    """Complete training pipeline with train/validation/test splits and regularization"""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE MODEL TRAINING PIPELINE WITH REGULARIZATION")
    logger.info("=" * 80)
    logger.info("Using 60%/20%/20% train/validation/test split")
    logger.info("Applying L2 regularization, dropout, and early stopping")
    logger.info("=" * 80)
    
    try:
        # Step 1: Preprocess data with proper splits
        logger.info("Step 1: Data preprocessing with train/val/test split...")
        from preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(logger=logger)
        df = preprocessor.load_and_preprocess(data_file)
        df = preprocessor.clean_and_transform_data(df)
        df = preprocessor.create_target_variable(df)
        df_encoded = preprocessor.encode_features(df)
        
        # Prepare features with train/validation/test split
        data_dict = preprocessor.prepare_features(df_encoded)
        if data_dict is None:
            logger.error("Failed to prepare features!")
            return None, None, None
        
        # Save preprocessors
        preprocessor.save_encoders()
        
        # Step 2: Initialize and build model with regularization
        logger.info("\nStep 2: Building model architecture with regularization...")
        model = MentalHealthModel(
            input_dim_behavioral=len(data_dict['behavioral_feature_names']),
            input_dim_demographic=len(data_dict['demographic_feature_names']),
            logger=logger
        )
        
        # Build autoencoder with L2 regularization
        if not model.build_autoencoder():
            logger.error("Failed to build autoencoder!")
            return None, None, None
        
        # Train autoencoder with train/validation/test split
        logger.info("\nStep 3: Training autoencoder with validation...")
        history = model.train_autoencoder(
            data_dict['X_beh_train'],
            data_dict['X_beh_val'],
            data_dict['X_beh_test'],
            epochs=epochs,
            batch_size=batch_size
        )
        
        if history is None:
            logger.error("Autoencoder training failed!")
            return None, None, None
        
        # Extract latent features for all sets
        logger.info("\nStep 4: Extracting latent features...")
        X_beh_train_latent = model.extract_features(data_dict['X_beh_train'])
        X_beh_val_latent = model.extract_features(data_dict['X_beh_val'])
        X_beh_test_latent = model.extract_features(data_dict['X_beh_test'])
        
        # Train hybrid model with validation
        logger.info("\nStep 5: Training hybrid model with validation and regularization...")
        X_combined = model.build_hybrid_model(
            X_beh_train_latent,
            data_dict['X_demo_train'],
            data_dict['y_train'],
            X_beh_val_latent,
            data_dict['X_demo_val'],
            data_dict['y_val']
        )
        
        if X_combined is None:
            logger.error("Hybrid model training failed!")
            return None, None, None
        
        # Evaluate model on both validation and test sets
        logger.info("\nStep 6: Evaluating model on validation and test sets...")
        results = model.evaluate_model(
            data_dict['X_beh_test'],
            data_dict['X_demo_test'],
            data_dict['y_test'],
            data_dict['X_beh_val'],
            data_dict['X_demo_val'],
            data_dict['y_val']
        )
        
        if results is None:
            logger.error("Model evaluation failed!")
            return None, None, None
        
        # Save models
        logger.info("\nStep 7: Saving models...")
        if not model.save_models():
            logger.error("Failed to save models!")
            return None, None, None
        
        # Final summary
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("FINAL RESULTS:")
        logger.info(f"  Validation Accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"  Accuracy Difference: {results['accuracy_diff']:.4f}")
        if results['accuracy_diff'] > 0.05:
            logger.warning("  ⚠️  Model may be overfitting!")
        else:
            logger.info("  ✅ Model shows good generalization!")
        logger.info("=" * 80)
        
        return model, results, data_dict
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return None, None, None