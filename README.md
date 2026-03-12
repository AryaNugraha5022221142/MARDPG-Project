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
- **Weights & Biases Integration**: Live tracking of training metrics, rewards, and success rates.

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

### 2. Test Classical Baseline (Potential Field)
Before running MARDPG, you can test a classical navigation algorithm:
```bash
python scripts/test_classical.py --scenario static_dense --render
```

### 3. Training
To start the training process:
```bash
python scripts/train.py --config config/config.yaml --run-name my_first_test
```

### 4. Evaluation
To test a trained model:
```bash
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/mardpg_final.pt --render
```

## 🌍 Environment Scenarios
You can test the algorithms in different scenarios by using the `--scenario` flag in the scripts:
- `empty`: No obstacles, pure navigation test.
- `static_dense`: 40 static obstacles, high density.
- `dynamic_chaos`: 20 obstacles, 80% are moving dynamically.
- `narrow_passage`: A real wall with a gap that agents must navigate through.
- `city`: **Urban Canyon** - High-rise buildings in a grid with No-Fly Zones and altitude limits.
- `warzone`: **Contested Airspace** - Terrain masking, Radar detection zones (yellow), and lethal Missile envelopes (red).
- `forest`: **Under-canopy** - Dense trunks (brown) and branches (green) requiring precise navigation.

## 🏗️ Environment Structure
- **State Space (28D)**: 25 rangefinder rays (5x5 grid) + 3D relative goal info.
- **Action Space (6D Discrete)**: Forward, Yaw Left/Right, Up, Down, Hover.
- **Dynamics**: Simplified 3D quadcopter physics with inertia and drag.

## ☁️ Google Colab Workflow

For heavy training, use Google Colab to leverage free GPU resources:

1. Open a new notebook in Colab.
2. Set Runtime Type to **T4 GPU**.
3. Mount Google Drive and clone this repo.
4. Run the training script without the `--render` flag for maximum speed.

## 📂 Project Structure

- `agents/`: MARDPG implementation and Neural Network architectures.
- `envs/`: Quadcopter dynamics and 3D environment logic.
- `config/`: Hyperparameter settings and environment configurations.
- `scripts/`: Entry points for training, evaluation, and analysis.
- `tests/`: Automated verification tests.

## 📄 License
This project is part of a Master's/Bachelor's Thesis. All rights reserved.
