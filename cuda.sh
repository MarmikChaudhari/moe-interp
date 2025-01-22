# Update the system and install necessary packages
echo "Updating system packages..."
apt update && apt upgrade -y
apt install wget -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt update
apt install cuda-toolkit-12-1 -y

# Install PyTorch with specific index URL
echo "Installing PyTorch..."
pip3 install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
echo "Installing Python dependencies..."
pip3 install kaleido transformers==4.44.2 tqdm pandas numpy accelerate

# set cuda environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# install flash-attn
pip3 install flash-attn --no-build-isolation

# install plotly matplotlib
pip3 install plotly matplotlib scikit-learn

echo "All dependencies installed successfully!"