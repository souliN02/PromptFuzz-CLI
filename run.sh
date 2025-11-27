#!/bin/bash
# PromptFuzz CLI Launcher
# Run this script to start the interactive menu

echo "Starting PromptFuzz Interactive Mode..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run setup first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e ."
    echo ""
    exit 1
fi

# Activate virtual environment and run
source .venv/bin/activate
python src/cli.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred. Check the output above for details."
    read -p "Press Enter to close..."
fi
