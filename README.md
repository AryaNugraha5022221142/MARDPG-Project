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
