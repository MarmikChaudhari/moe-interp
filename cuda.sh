# Update the system and install necessary packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install PyTorch with specific index URL
echo "Installing PyTorch..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
echo "Installing Python dependencies..."
pip3 install kaleido transformers tqdm pandas numpy

echo "All dependencies installed successfully!"