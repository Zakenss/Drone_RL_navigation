#!/usr/bin/env python3
"""
Main training script for drone RL agent.
Run with: python train_agent.py --config config/training.yaml
"""

import argparse
import yaml
import time
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train RL agent for drone navigation')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 60)
    logger.info("Starting RL training for drone navigation")
    logger.info("=" * 60)
    
    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import modules after arg parsing
    from src.environment.drone_env import UrbanDroneEnv
    from src.rl_agent.ppo_trainer import PPOTrainer
    from src.database.flight_logger import FlightLogger
    
    # Initialize components
    logger.info("Initializing environment and agent...")
    
    # Create environment
    env_config = config['environment']
    env = UrbanDroneEnv(env_config)
    
    # Create RL agent
    agent_config = config['agent']
    agent = PPOTrainer(agent_config)
    
    # Create flight logger for compliance tracking
    flight_logger = FlightLogger(config['database']['path'])
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)
    
    # Training loop
    logger.info(f"Starting training for {args.episodes} episodes...")
    
    best_success_rate = 0.0
    
    for episode in range(args.episodes):
        episode_start = time.time()
        
        # Start flight logging
        flight_id = flight_logger.start_flight({
            'weather': env.weather,
            'helicopter_speed': env.helicopter_speed
        })
        
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Log step data
            flight_logger.log_step(flight_id, {
                'step_number': step_count,
                'position': env.drone.position.tolist(),
                'altitude': float(env.drone.altitude),
                'velocity': env.drone.velocity.tolist(),
                'distance_to_helicopter': info['distance_to_helicopter'],
                'action': action,
                'reward': reward,
                'weather': env.weather,
                'helicopter_speed': env.helicopter_speed,
                'safety_violation': info.get('safety_violation', False),
                'altitude_violation': info.get('altitude_violation', False)
            })
            
            # Store transition in agent's memory
            agent.store_transition(state, action, reward, done, next_state)
            
            # Update for next step
            state = next_state
            total_reward += reward
            step_count += 1
        
        # End flight logging
        flight_logger.end_flight(flight_id, {
            'total_reward': total_reward,
            'collision_count': env.collision_count
        })
        
        # Update agent policy
        if len(agent.memory) >= agent.batch_size:
            agent.update_policy()
        
        # Calculate success rate for this episode
        flight_report = flight_logger.get_flight_report(flight_id)
        success_rate = flight_report['success_rate']
        
        # Update curriculum if needed
        curriculum_update = agent.update_curriculum(success_rate)
        if curriculum_update:
            env.update_parameters(curriculum_update)
        
        # Save best model
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            model_path = output_dir / f"best_model_{success_rate:.2f}.pt"
            agent.save_checkpoint(model_path, episode)
            logger.info(f"New best model saved: {model_path}")
        
        # Periodic checkpoint
        if episode % 100 == 0:
            checkpoint_path = output_dir / f"checkpoint_ep{episode}.pt"
            agent.save_checkpoint(checkpoint_path, episode)
        
        # Log progress
        episode_time = time.time() - episode_start
        
        if episode % 10 == 0:
            logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Steps: {step_count:3d} | "
                f"Success: {success_rate:.2f} | "
                f"Stage: {agent.current_stage + 1}/4 | "
                f"Time: {episode_time:.1f}s"
            )
    
    # Final save
    final_path = output_dir / "final_model.pt"
    agent.save_checkpoint(final_path, args.episodes)
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best success rate: {best_success_rate:.2f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 60)
    
    # Close resources
    env.close()
    flight_logger.close()

if __name__ == "__main__":
    main()