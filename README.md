# Autonomous Drone Navigation for Urban Logistics

## Project Overview
This repository contains the implementation of a reinforcement learning-based navigation system for autonomous drone delivery in urban environments. The system was developed as part of my research at UWE Bristol, focusing on safe navigation around dynamic obstacles like helicopters in varying weather conditions.

## Key Features
- PPO with LSTM for adaptive obstacle avoidance
- Curriculum learning from 15m/s to 25m/s helicopter speeds
- Weather-aware navigation (dry vs 8mm/hr rain)
- Camera-Radar sensor fusion for robust perception
- SQL-based flight logging for regulatory compliance
- Integration with Unreal Engine 4.27 and PX4


```bash
# Install dependencies
pip install -r requirements.txt

# Train the RL agent
python scripts/train_agent.py --config config/training.yaml

# Run simulation
python scripts/run_simulation.py --scenario urban --weather rain
