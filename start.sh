#!/bin/bash
echo "starting AI for macOs / Linux"


# Try running the Python script with 'python'
if command -v python &> /dev/null; then
    echo "Using 'python' to run the script..."
    python index.py
    if [ $? -ne 0 ]; then
        echo "'python' failed. Trying 'python3'..."
        python3 index.py
    fi
else
    # If 'python' is not available, try 'python3'
    echo "'python' is not available. Trying 'python3'..."
    python3 index.py
fi
