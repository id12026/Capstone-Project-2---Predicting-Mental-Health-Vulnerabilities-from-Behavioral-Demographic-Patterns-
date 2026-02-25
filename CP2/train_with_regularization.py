#!/usr/bin/env python3
"""
Training script with balanced regularization to achieve target accuracy of 83-86%
"""

import os
import sys
import argparse
from datetime import datetime

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import EnhancedLogger, logger
from model_training import train_complete_model

def setup_environment():
    """Setup environment and logging"""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('logs/autoencoder_tensorboard', exist_ok=True)
    
    # Setup enhanced logger
    enhanced_logger = EnhancedLogger("regularized_training")
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
    """Main execution function with regularization"""
    parser = argparse.ArgumentParser(
        description='Mental Health Prediction Training with Regularization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --data data/mental_health_data.csv     # Train with default settings
  %(prog)s --epochs 100 --batch-size 16           # Train with custom parameters
  %(prog)s --data custom_data.csv                 # Train with custom data file
        '''
    )
    
    parser.add_argument('--data', 
                       default='data/mental_health_data.csv', 
                       help='Path to dataset')
    parser.add_argument('--epochs', 
                       default=50, 
                       type=int, 
                       help='Training epochs for autoencoder')
    parser.add_argument('--batch-size', 
                       default=32, 
                       type=int, 
                       help='Training batch size')
    parser.add_argument('--learning-rate', 
                       default=0.0005, 
                       type=float, 
                       help='Learning rate for autoencoder')
    
    args = parser.parse_args()
    
    # Setup environment and logging
    enhanced_logger = setup_environment()
    log_file = enhanced_logger.get_log_file()
    
    logger.info("=" * 80)
    logger.info("MENTAL HEALTH VULNERABILITY PREDICTION - REGULARIZED TRAINING")
    logger.info("=" * 80)
    logger.info("Balanced Regularization Training Pipeline with:")
    logger.info("  • 60%/20%/20% train/validation/test split")
    logger.info("  • Moderate L2 regularization (0.005)")
    logger.info("  • Moderate dropout (0.3)")
    logger.info("  • Balanced early stopping with patience=12")
    logger.info("  • Balanced learning rate scheduling")
    logger.info("  • XGBoost with moderate regularization")
    logger.info("=" * 80)
    logger.info(f"Data file: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    
    try:
        # Check data file
        data_file = check_and_fix_data(args.data, logger)
        if not data_file:
            logger.error("No valid data file available. Exiting.")
            sys.exit(1)
        
        logger.info(f"Using data file: {data_file}")
        
        # Start training with regularization
        logger.info("Starting regularized training pipeline...")
        model, results, data_dict = train_complete_model(
            data_file, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            logger=logger
        )
        
        if model and results:
            logger.info("=" * 80)
            logger.info("REGULARIZED TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info("FINAL PERFORMANCE METRICS:")
            logger.info(f"  Validation Accuracy: {results['val_accuracy']:.4f}")
            logger.info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            logger.info(f"  Validation Precision: {results['val_precision']:.4f}")
            logger.info(f"  Test Precision: {results['test_precision']:.4f}")
            logger.info(f"  Validation Recall: {results['val_recall']:.4f}")
            logger.info(f"  Test Recall: {results['test_recall']:.4f}")
            logger.info(f"  Validation F1-Score: {results['val_f1']:.4f}")
            logger.info(f"  Test F1-Score: {results['test_f1']:.4f}")
            logger.info("=" * 80)
            logger.info("OVERFITTING ANALYSIS:")
            logger.info(f"  Accuracy Difference: {results['accuracy_diff']:.4f}")
            if results['accuracy_diff'] > 0.05:
                logger.warning("  ⚠️  WARNING: Significant overfitting detected!")
                logger.warning("  Consider: more regularization, more data, or simpler model")
            elif results['accuracy_diff'] > 0.02:
                logger.info("  ⚠️  Mild overfitting - monitor performance")
            else:
                logger.info("  ✅ Good generalization - model performs consistently")
            logger.info("=" * 80)
            logger.info("Models saved in: models/")
            logger.info("Training logs available in: logs/")
            logger.info("=" * 80)
        else:
            logger.error("Training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING INTERRUPTED BY USER")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error during training: {e}")
        enhanced_logger.log_exception(e, "Regularized training")
        sys.exit(1)
        
    finally:
        logger.info("=" * 80)
        logger.info(f"TRAINING SESSION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file saved: {log_file}")
        logger.info("=" * 80)

if __name__ == '__main__':
    main()