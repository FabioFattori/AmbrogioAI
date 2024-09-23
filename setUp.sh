echo "Setting up the environment..."
#!/bin/bash

# Check if Python is installed
if command -v python3 &>/dev/null; then
    echo "Python is already installed"
else
    echo "Python is not installed. Installing Python..."
    
    # Update package list
    sudo apt update
    
    # Install Python
    sudo apt install -y python3
    
    echo "Python installed successfully"
fi

# Check if pip3 is installed
if command -v pip3 &> /dev/null; then
    echo "pip is already installed"
else
    echo "pip is not installed. Installing pip..."
    sudo apt update
    sudo apt install -y python3-pip
    echo "pip installed successfully"
fi

cd script
./installDependecies.sh
cd ..
echo "If you see no errors, the environment is set up correctly."
echo "You can now run the program by executing the start.sh script."