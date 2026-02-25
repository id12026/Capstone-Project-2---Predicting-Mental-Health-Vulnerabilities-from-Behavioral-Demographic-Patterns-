#!/bin/bash

# Mental Health Prediction Project - Complete Setup Script
# Author: System
# Date: $(date)

echo "==============================================="
echo "   MENTAL HEALTH PREDICTION SYSTEM"
echo "   Complete Setup and Execution Script"
echo "==============================================="
echo ""

# Set up variables
PROJECT_DIR="$(pwd)"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/setup_$TIMESTAMP.log"

# Create directory structure
echo "📁 Creating project structure..."
mkdir -p $PROJECT_DIR/{data,models,logs,static/{css,js},templates,utils} 2>&1 | tee -a $MAIN_LOG

# Check Python version
echo "🐍 Checking Python version..."
python3 --version 2>&1 | tee -a $MAIN_LOG

# Create virtual environment
echo "🔧 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv 2>&1 | tee -a $MAIN_LOG
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/Scripts/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip 2>&1 | tee -a $MAIN_LOG

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt 2>&1 | tee -a $MAIN_LOG
    echo "✅ Requirements installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Check data
echo "📊 Checking data..."
if [ -f "data/mental_health_data.csv" ]; then
    echo "✅ Data file found"
    DATA_SIZE=$(wc -l < data/mental_health_data.csv)
    echo "   Records: $DATA_SIZE"
else
    echo "⚠️  No data file found. Creating test data..."
    python3 create_test_data.py 2>&1 | tee -a $MAIN_LOG
fi

# Display menu
echo ""
echo "==============================================="
echo "   SELECT OPERATION MODE"
echo "==============================================="
echo "1. Train Model"
echo "2. Start Web Server"
echo "3. Test System"
echo "4. Diagnose Issues"
echo "5. Quick Fix"
echo "6. View Logs"
echo "7. Exit"
echo "==============================================="
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo "🤖 Starting model training..."
        python3 main.py --mode train --epochs 30 --batch-size 32 2>&1 | tee -a $MAIN_LOG
        ;;
    2)
        echo "🌐 Starting web server..."
        echo "   Open http://localhost:5000 in your browser"
        echo "   Press Ctrl+C to stop the server"
        echo "   Logs will be saved to logs/flask_app_*.log"
        echo ""
        python3 main.py --mode serve --port 5000 --debug 2>&1 | tee -a $MAIN_LOG
        ;;
    3)
        echo "🧪 Testing system..."
        python3 main.py --mode test 2>&1 | tee -a $MAIN_LOG
        ;;
    4)
        echo "🔍 Diagnosing issues..."
        python3 main.py --mode diagnose 2>&1 | tee -a $MAIN_LOG
        ;;
    5)
        echo "🛠️  Applying quick fix..."
        python3 quick_fix.py 2>&1 | tee -a $MAIN_LOG
        ;;
    6)
        echo "📋 Viewing logs..."
        if [ -d "logs" ]; then
            echo "Latest log files:"
            ls -la logs/*.log | tail -5
            echo ""
            read -p "Enter log filename to view (or press Enter for latest): " logfile
            if [ -z "$logfile" ]; then
                logfile=$(ls -t logs/*.log | head -1)
            fi
            if [ -f "$logfile" ]; then
                echo "Last 50 lines of $logfile:"
                echo "==============================================="
                tail -50 "$logfile"
                echo "==============================================="
            else
                echo "❌ Log file not found!"
            fi
        else
            echo "❌ No logs directory found!"
        fi
        ;;
    7)
        echo "👋 Exiting..."
        exit 0
        ;;
    *)
        echo "❌ Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "==============================================="
echo "   OPERATION COMPLETED"
echo "==============================================="
echo "Main log: $MAIN_LOG"
echo "All logs: $LOG_DIR/"
echo "==============================================="