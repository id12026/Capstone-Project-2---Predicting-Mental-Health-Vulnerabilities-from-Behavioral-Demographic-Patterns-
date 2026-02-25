#!/usr/bin/env python3
"""
Enhanced logging utility with file logging and rotation
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

class EnhancedLogger:
    """Enhanced logger with file and console output"""
    
    _instance = None
    
    def __new__(cls, log_name="mental_health_app"):
        if cls._instance is None:
            cls._instance = super(EnhancedLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_name="mental_health_app"):
        if self._initialized:
            return
        
        self.log_name = log_name
        self.log_dir = "logs"
        self.setup_logging()
        self._initialized = True
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate unique log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.log_dir}/{self.log_name}_{timestamp}.log"
        
        # Configure root logger
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed)
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (simple)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log startup information
        self.logger.info("=" * 60)
        self.logger.info(f"LOG SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log file: {os.path.abspath(log_filename)}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info("=" * 60)
        
        self.current_log_file = log_filename
    
    def get_logger(self):
        """Get the logger instance"""
        return self.logger
    
    def get_log_file(self):
        """Get current log file path"""
        return self.current_log_file
    
    def log_system_info(self):
        """Log system information"""
        import platform
        import psutil
        
        self.logger.info("=" * 60)
        self.logger.info("SYSTEM INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Processor: {platform.processor()}")
        self.logger.info(f"Python: {platform.python_version()}")
        self.logger.info(f"CPU Cores: {psutil.cpu_count(logical=True)}")
        self.logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        self.logger.info(f"Disk free: {psutil.disk_usage('.').free / (1024**3):.2f} GB")
        self.logger.info("=" * 60)
    
    def log_package_versions(self):
        """Log installed package versions"""
        import pkg_resources
        
        self.logger.info("=" * 60)
        self.logger.info("PACKAGE VERSIONS")
        self.logger.info("=" * 60)
        
        packages = [
            'pandas', 'numpy', 'scikit-learn', 'tensorflow',
            'xgboost', 'shap', 'flask', 'matplotlib', 'seaborn'
        ]
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                self.logger.info(f"{package:20} v{version}")
            except:
                self.logger.warning(f"{package:20} Not installed")
        
        self.logger.info("=" * 60)
    
    def log_exception(self, e, context=""):
        """Log exception with context"""
        self.logger.error("=" * 60)
        self.logger.error(f"EXCEPTION OCCURRED: {context}")
        self.logger.error("=" * 60)
        self.logger.error(f"Exception type: {type(e).__name__}")
        self.logger.error(f"Exception message: {str(e)}")
        import traceback
        self.logger.error("Traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.logger.error(f"  {line}")
        self.logger.error("=" * 60)
    
    def log_prediction(self, form_data, prediction_result):
        """Log prediction details"""
        self.logger.info("=" * 60)
        self.logger.info("PREDICTION MADE")
        self.logger.info("=" * 60)
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("Input data:")
        for key, value in form_data.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"Prediction: {prediction_result.get('risk_level', 'Unknown')}")
        self.logger.info(f"Confidence: {prediction_result.get('confidence', 0):.2f}%")
        self.logger.info(f"Behavioral Score: {prediction_result.get('behavioral_score', 0)}")
        self.logger.info("=" * 60)

# Global logger instance
logger = EnhancedLogger().get_logger()