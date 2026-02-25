#!/usr/bin/env python3
"""
Main execution script for Mental Health Vulnerability Prediction System
with enhanced logging
"""

import os
import sys
import argparse
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import EnhancedLogger, logger

def setup_environment():
    """Setup environment and logging"""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Setup enhanced logger
    enhanced_logger = EnhancedLogger("mental_health_system")
    global logger
    logger = enhanced_logger.get_logger()
    
    # Log system info
    enhanced_logger.log_system_info()
    enhanced_logger.log_package_versions()
    
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    return enhanced_logger

def check_and_fix_data(data_file, logger):
    """Check if data file exists and is valid"""
    logger.info(f"Checking data file: {data_file}")
    
    if not os.path.exists(data_file):
        logger.warning(f"Data file not found: {data_file}")
        logger.info("Creating test dataset...")
        
        try:
            from create_test_data import create_test_dataset
            new_file = create_test_dataset()
            logger.info(f"Test dataset created: {new_file}")
            return new_file
        except Exception as e:
            logger.error(f"Failed to create test dataset: {e}")
            return None
    
    # Check file size
    file_size = os.path.getsize(data_file)
    logger.info(f"Data file size: {file_size / 1024:.2f} KB")
    
    if file_size == 0:
        logger.error("Data file is empty!")
        return None
    
    return data_file

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Mental Health Vulnerability Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --mode train           # Train the model
  %(prog)s --mode serve           # Start web server
  %(prog)s --mode test            # Test data loading
  %(prog)s --mode explain         # Generate explainability analysis
  %(prog)s --mode diagnose        # Diagnose issues
        '''
    )
    
    parser.add_argument('--mode', 
                       choices=['train', 'predict', 'explain', 'serve', 'test', 'diagnose', 'quickfix'], 
                       default='serve', 
                       help='Operation mode')
    parser.add_argument('--data', 
                       default='data/mental_health_data.csv', 
                       help='Path to dataset')
    parser.add_argument('--port', 
                       default=5000, 
                       type=int, 
                       help='Port for Flask server')
    parser.add_argument('--epochs', 
                       default=50, 
                       type=int, 
                       help='Training epochs')
    parser.add_argument('--batch-size', 
                       default=32, 
                       type=int, 
                       help='Training batch size')
    parser.add_argument('--debug', 
                       action='store_true', 
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup environment and logging
    enhanced_logger = setup_environment()
    log_file = enhanced_logger.get_log_file()
    
    logger.info("=" * 80)
    logger.info("MENTAL HEALTH VULNERABILITY PREDICTION SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    try:
        if args.mode == 'train':
            logger.info("Starting training mode...")
            from model_training import train_complete_model
            
            data_file = check_and_fix_data(args.data, logger)
            if not data_file:
                logger.error("No valid data file available. Exiting.")
                sys.exit(1)
            
            logger.info(f"Using data file: {data_file}")
            logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
            
            model, results, data_dict = train_complete_model(
                data_file, 
                epochs=args.epochs, 
                batch_size=args.batch_size,
                logger=logger
            )
            
            if model and results:
                logger.info("=" * 60)
                logger.info("TRAINING COMPLETED SUCCESSFULLY!")
                logger.info("=" * 60)
                logger.info(f"Model Accuracy: {results['accuracy']:.4f}")
                logger.info(f"Precision: {results['precision']:.4f}")
                logger.info(f"Recall: {results['recall']:.4f}")
                logger.info(f"F1-Score: {results['f1']:.4f}")
                logger.info(f"Models saved in: models/")
                logger.info("=" * 60)
            else:
                logger.error("Training failed!")
                
        elif args.mode == 'test':
            logger.info("Starting test mode...")
            from preprocessing import DataPreprocessor
            
            data_file = check_and_fix_data(args.data, logger)
            if not data_file:
                logger.error("No valid data file available. Exiting.")
                sys.exit(1)
            
            preprocessor = DataPreprocessor(logger=logger)
            df = preprocessor.load_and_preprocess(data_file)
            
            if df is not None:
                logger.info("=" * 60)
                logger.info("DATA TEST COMPLETED SUCCESSFULLY!")
                logger.info("=" * 60)
                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                logger.info(f"Data types:\n{df.dtypes}")
                logger.info(f"Missing values:\n{df.isnull().sum()}")
                logger.info("=" * 60)
                
                # Test target creation
                df = preprocessor.clean_and_transform_data(df)
                df_target = preprocessor.create_target_variable(df)
                
                if 'risk_level' in df_target.columns:
                    logger.info(f"Target variable created successfully!")
                    logger.info(f"Risk level distribution:\n{df_target['risk_level'].value_counts()}")
                else:
                    logger.error("Failed to create target variable")
            else:
                logger.error("Data loading failed!")
                
        elif args.mode == 'explain':
            logger.info("Starting explainability mode...")
            from explainability import run_explainability_analysis
            from model_training import MentalHealthModel
            from preprocessing import DataPreprocessor
            
            try:
                model = MentalHealthModel(logger=logger)
                model.load_models()
                logger.info("Models loaded successfully")
                
                data_file = check_and_fix_data(args.data, logger)
                if not data_file:
                    logger.error("No valid data file available. Exiting.")
                    sys.exit(1)
                
                preprocessor = DataPreprocessor(logger=logger)
                df = preprocessor.load_and_preprocess(data_file)
                df = preprocessor.clean_and_transform_data(df)
                df_target = preprocessor.create_target_variable(df)
                df_encoded = preprocessor.encode_features(df_target)
                data_dict = preprocessor.prepare_features(df_encoded)
                
                if data_dict:
                    logger.info("=" * 60)
                    logger.info("EXPLAINABILITY ANALYSIS SKIPPED - REMOVED")
                    logger.info("=" * 60)
                    logger.info("SHAP-based explainability has been removed from this application")
                    logger.info("=" * 60)
                else:
                    logger.error("Failed to prepare features for explainability")
                    
            except Exception as e:
                logger.error(f"Explainability analysis failed: {e}")
                enhanced_logger.log_exception(e, "Explainability analysis")
                
        elif args.mode == 'diagnose':
            logger.info("Starting diagnostic mode...")
            from diagnose_issue import diagnose_feature_mismatch
            diagnose_feature_mismatch(logger)
            
        elif args.mode == 'quickfix':
            logger.info("Starting quick fix mode...")
            from quick_fix import quick_fix
            quick_fix(logger)
            
        elif args.mode in ['serve', 'predict']:
            logger.info("Starting web server mode...")
            from app import app, load_models
            
            if load_models(logger):
                logger.info("=" * 60)
                logger.info("WEB SERVER STARTING")
                logger.info("=" * 60)
                logger.info(f"Server URL: http://localhost:{args.port}")
                logger.info(f"Dashboard: http://localhost:{args.port}/dashboard")
                logger.info(f"API Endpoint: http://localhost:{args.port}/predict")
                logger.info("=" * 60)
                logger.info("Press Ctrl+C to stop the server")
                logger.info("=" * 60)
                
                app.run(
                    debug=args.debug, 
                    port=args.port, 
                    use_reloader=False,
                    host='0.0.0.0'
                )
            else:
                logger.error("Failed to load models. Please train first.")
                logger.info("Try: python main.py --mode train")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("PROGRAM INTERRUPTED BY USER")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error in mode '{args.mode}': {e}")
        enhanced_logger.log_exception(e, f"Mode {args.mode}")
        sys.exit(1)
        
    finally:
        logger.info("=" * 80)
        logger.info(f"SESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file saved: {log_file}")
        logger.info("=" * 80)

if __name__ == '__main__':
    main()