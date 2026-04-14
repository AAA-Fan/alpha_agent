#!/bin/bash

# Multi-Agent Financial Advisory System - Setup Script

echo "=========================================="
echo "Financial Advisory System - Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate the virtual environment:"
    echo "  source venv/bin/activate  (Linux/Mac)"
    echo "  venv\\Scripts\\activate     (Windows)"
    echo ""
    read -p "Press Enter to continue after activating the virtual environment..."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

echo ""

# Setup .env file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "IMPORTANT: Please edit .env and add your OpenAI API key:"
    echo "  OPENAI_API_KEY=your_api_key_here"
    echo ""
    read -p "Press Enter to continue..."
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. (Optional) Add Tavily API key for enhanced news search"
echo "3. Run the system: python main.py"
echo ""
echo "Example usage: python example_usage.py"
echo ""
