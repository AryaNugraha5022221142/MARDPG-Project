# MARDPG Quadcopter Thesis: Multi-Agent Reinforcement Learning for UAV Navigation

This repository contains the implementation of a Multi-Agent Reinforcement Learning (MARL) framework based on the **Multi-Agent Recurrent Deterministic Policy Gradient (MARDPG)** algorithm. The project focuses on autonomous navigation of multiple quadcopters in a complex, 3D environment filled with static and dynamic obstacles.

## 🚀 Project Overview

The goal of this project is to train multiple UAVs (quadcopters) to navigate from starting positions to specific goals while:
- Avoiding collisions with **static obstacles** (spheres and buildings).
- Avoiding collisions with **dynamic obstacles** (moving objects).
- Maintaining safe distances from other agents.
- Optimizing flight paths for efficiency.

## 🛠 Features

- **MARDPG Algorithm**: A state-of-the-art MARL algorithm using centralized training and decentralized execution with LSTM-based recurrent networks.
- **Complex 3D Environment**: A custom environment built with realistic quadcopter dynamics and varied obstacle shapes (spheres and boxes).
- **Dynamic Obstacles**: Real-time moving threats to test the agents' predictive capabilities.
- **3D Visualization**: Real-time rendering using Matplotlib for monitoring agent behavior.

## 📦 Installation

### Local Setup (VS Code)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Login to WandB**:
   ```bash
   wandb login
   ```

## 🏃 Usage

### 1. Visualize the Environment
To see the environment scenarios without training:
```bash
python scripts/visualize_env.py --scenario static_dense
```

### 2. Training
To start the training process:
```bash
python scripts/train.py --config config/config.yaml --run-name my_first_test
```

### 3. Evaluation
To test a trained model:
```bash
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/mardpg_final.pt --render
```


## 📄 License
This project is part of a Master's/Bachelor's Thesis. All rights reserved.
