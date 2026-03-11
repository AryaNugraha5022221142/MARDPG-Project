# Colab Setup Helper
# Run this in a Colab cell to quickly set up the environment

import os

def setup_colab(repo_url):
    # 1. Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # 2. Clone the repo
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(repo_name):
        print(f"Cloning {repo_url}...")
        os.system(f"git clone {repo_url}")
    
    os.chdir(repo_name)
    
    # 3. Install requirements
    print("Installing requirements...")
    os.system("pip install -r requirements.txt")
    
    # 4. Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\nSetup Complete!")
    print(f"Current directory: {os.getcwd()}")
    print("You can now run: !python scripts/train.py --config config/config.yaml")

# Example usage:
# setup_colab("https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git")
