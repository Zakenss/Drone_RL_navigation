"""
Urban drone navigation environment with dynamic obstacles.
Implements the Gym interface for RL training.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import random
from datetime import datetime

@dataclass
class DroneState:
    """Current state of the drone for navigation."""
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    altitude: float  # meters above ground
    payload_mass: float = 8.0  # default 8kg for delivery
    
    @property
    def speed(self) -> float:
        """Get horizontal speed magnitude."""
        return np.linalg.norm(self.velocity[:2])

class UrbanDroneEnv:
    """Main environment class for drone navigation in Bristol city centre."""
    
    def __init__(self, config: Dict):
        # Environment parameters from config
        self.city_centre_bounds = config.get('city_bounds', {
            'x_min': -500, 'x_max': 500,
            'y_min': -500, 'y_max': 500,
            'z_min': 0, 'z_max': 150
        })
        
        # Helicopter parameters
        self.helicopter_speed = config.get('helicopter_speed', 15.0)  # m/s
        self.helicopter_altitude = config.get('helicopter_altitude', 100.0)
        self.helicopter_path = self._generate_helipath()
        
        # Weather effects
        self.weather = config.get('weather', 'dry')
        self.rain_intensity = config.get('rain_intensity', 0.0)  # mm/hr
        self.braking_penalty = 0.6 if self.weather == 'rain' else 1.0
        
        # Safety parameters
        self.safety_margin = 15.0  # meters
        self.altitude_limit = 120.0  # UK CAA regulation
        self.min_altitude = 10.0  # avoid ground
        
        # Drone initialization
        self.drone = DroneState(
            position=np.array([0, 0, 100.0]),
            velocity=np.array([8.33, 0, 0]),  # 30 km/h cruising
            orientation=np.zeros(3),
            altitude=100.0
        )
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        self.total_reward = 0.0
        self.collision_count = 0
        
        # Observation space setup
        self.observation_shape = (84, 84, 3)  # RGB camera input
        self.state_dim = 10  # [pos(3), vel(3), orientation(3), altitude]
        
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment for new episode."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset drone to starting position
        self.drone = DroneState(
            position=np.array([0, 0, 100.0]),
            velocity=np.array([8.33, 0, 0]),
            orientation=np.zeros(3),
            altitude=100.0
        )
        
        # Reset helicopter path
        self.helicopter_position = np.array([100.0, 0, 100.0])
        self.helicopter_velocity = np.array([-self.helicopter_speed, 0, 0])
        
        # Reset tracking variables
        self.step_count = 0
        self.total_reward = 0.0
        self.collision_count = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """Execute one timestep with given action."""
        self.step_count += 1
        
        # Parse action into control commands
        control = self._parse_action(action)
        
        # Apply control with physics model
        self._apply_control(control)
        
        # Update helicopter position
        self._update_helicopter()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination conditions
        done = self._check_termination()
        
        # Get next observation
        observation = self._get_observation()
        
        # Collect info for debugging
        info = {
            'distance_to_helicopter': self._get_distance_to_helicopter(),
            'altitude': float(self.drone.altitude),
            'speed': float(self.drone.speed),
            'weather': self.weather,
            'step': self.step_count
        }
        
        return observation, reward, done, info
    
    def _parse_action(self, action: np.ndarray) -> Dict:
        """Convert normalized RL action to drone control commands."""
        # Action space: [thrust, pitch, roll, yaw] in range [-1, 1]
        thrust = 500 + 100 * np.clip(action[0], -1, 1)  # PWM 400-600
        pitch = 25 * np.clip(action[1], -1, 1)  # degrees
        roll = 25 * np.clip(action[2], -1, 1)   # degrees
        yaw = 180 * np.clip(action[3], -1, 1)   # degrees
        
        # Apply weather effects on thrust
        if self.weather == 'rain':
            thrust *= self.braking_penalty
        
        return {
            'thrust': thrust,
            'pitch': np.deg2rad(pitch),
            'roll': np.deg2rad(roll),
            'yaw': np.deg2rad(yaw)
        }
    
    def _apply_control(self, control: Dict):
        """Update drone state based on control inputs."""
        # Simplified quadcopter dynamics
        dt = 0.1  # 10Hz control frequency
        
        # Update position based on velocity
        self.drone.position += self.drone.velocity * dt
        
        # Update velocity based on thrust and orientation
        acceleration = np.array([
            control['thrust'] * np.cos(control['pitch']) * np.cos(control['yaw']),
            control['thrust'] * np.sin(control['roll']) * np.cos(control['pitch']),
            control['thrust'] * np.sin(control['pitch'])
        ])
        
        # Apply drag (simplified)
        drag_coefficient = 0.1
        drag = -drag_coefficient * self.drone.velocity
        
        # Update velocity
        self.drone.velocity += (acceleration + drag) * dt
        
        # Update orientation
        self.drone.orientation = np.array([
            control['roll'],
            control['pitch'],
            control['yaw']
        ])
        
        # Update altitude (z-coordinate)
        self.drone.altitude = self.drone.position[2]
        
        # Add noise for realism
        self._add_sensor_noise()
    
    def _update_helicopter(self):
        """Move helicopter along predefined path."""
        dt = 0.1
        self.helicopter_position += self.helicopter_velocity * dt
        
        # Simple boundary check - reverse direction if out of bounds
        bounds = self.city_centre_bounds
        if (self.helicopter_position[0] < bounds['x_min'] or 
            self.helicopter_position[0] > bounds['x_max']):
            self.helicopter_velocity[0] *= -1
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on safety and performance."""
        reward = 0.0
        distance = self._get_distance_to_helicopter()
        
        # Collision penalty (severe)
        if distance < 2.0:
            reward -= 100.0
            self.collision_count += 1
        
        # Safety margin penalty (gradual)
        elif distance < self.safety_margin:
            penalty = 10.0 * (self.safety_margin - distance)
            reward -= penalty
        
        # Altitude compliance
        if self.drone.altitude > self.altitude_limit:
            reward -= 50.0
        elif self.drone.altitude < self.min_altitude:
            reward -= 50.0
        
        # Speed maintenance bonus
        if 7.5 < self.drone.speed < 9.0:
            reward += 0.1
        
        # Survival bonus (encourage longer episodes)
        reward += 0.01
        
        # Smooth flight bonus (small orientation changes)
        orientation_change = np.linalg.norm(self.drone.orientation)
        if orientation_change < 0.1:
            reward += 0.05
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should end."""
        # Collision with helicopter
        if self._get_distance_to_helicopter() < 1.0:
            return True
        
        # Altitude violations
        if (self.drone.altitude > self.altitude_limit + 5.0 or 
            self.drone.altitude < self.min_altitude - 2.0):
            return True
        
        # Out of bounds
        bounds = self.city_centre_bounds
        pos = self.drone.position
        if (pos[0] < bounds['x_min'] or pos[0] > bounds['x_max'] or
            pos[1] < bounds['y_min'] or pos[1] > bounds['y_max']):
            return True
        
        # Maximum steps reached
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def _get_observation(self) -> Dict:
        """Construct observation from sensors."""
        # Camera observation (simulated)
        camera_img = self._render_camera_view()
        
        # Radar observation (simulated)
        radar_data = self._get_radar_measurement()
        
        # State vector
        state_vector = np.concatenate([
            self.drone.position,
            self.drone.velocity,
            self.drone.orientation,
            [self.drone.altitude]
        ])
        
        return {
            'camera': camera_img,
            'radar': radar_data,
            'state': state_vector,
            'distance': self._get_distance_to_helicopter()
        }
    
    def _render_camera_view(self) -> np.ndarray:
        """Generate synthetic camera image with helicopter."""
        img = np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Add background (sky)
        img[:, :] = [135, 206, 235]  # Light blue
        
        # Calculate helicopter position in image coordinates
        rel_pos = self.helicopter_position - self.drone.position
        scale = 5.0  # pixels per meter
        
        # Convert to image coordinates (centered)
        img_center = np.array([42, 42])
        heli_pixel = img_center + rel_pos[:2] * scale
        
        # Draw helicopter if in view
        if (0 <= heli_pixel[0] < 84 and 0 <= heli_pixel[1] < 84):
            # Helicopter as red circle
            cv2.circle(img, 
                      (int(heli_pixel[0]), int(heli_pixel[1])), 
                      3, (0, 0, 255), -1)
        
        # Add rain effects if applicable
        if self.weather == 'rain':
            img = self._add_rain_effects(img)
        
        return img
    
    def _add_rain_effects(self, img: np.ndarray) -> np.ndarray:
        """Add visual rain effects to camera image."""
        # Reduce contrast
        img = cv2.addWeighted(img, 0.7, np.zeros_like(img), 0.3, 0)
        
        # Add rain streaks
        num_drops = int(self.rain_intensity * 0.2)
        for _ in range(num_drops):
            x = random.randint(0, 83)
            length = random.randint(3, 8)
            cv2.line(img, (x, 0), (x, length), (200, 200, 200), 1)
        
        # Add Gaussian blur to simulate water droplets
        if self.rain_intensity > 4.0:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def _get_radar_measurement(self) -> np.ndarray:
        """Simulate radar readings for obstacle detection."""
        # Simple radar: returns distance and relative velocity
        distance = self._get_distance_to_helicopter()
        rel_velocity = self.helicopter_velocity - self.drone.velocity
        
        # Add noise based on weather
        noise_level = 0.1 if self.weather == 'dry' else 0.3
        distance_noise = random.gauss(0, noise_level * distance)
        velocity_noise = random.gauss(0, noise_level * 2)
        
        return np.array([
            distance + distance_noise,
            rel_velocity[0] + velocity_noise,
            rel_velocity[1] + velocity_noise
        ])
    
    def _get_distance_to_helicopter(self) -> float:
        """Calculate Euclidean distance to helicopter."""
        return np.linalg.norm(self.drone.position - self.helicopter_position)
    
    def _add_sensor_noise(self):
        """Add realistic sensor noise to drone state."""
        noise_scale = 0.01  # 1% noise
        
        # Add Gaussian noise to position
        pos_noise = np.random.randn(3) * noise_scale * 10
        self.drone.position += pos_noise
        
        # Add Gaussian noise to velocity
        vel_noise = np.random.randn(3) * noise_scale * 2
        self.drone.velocity += vel_noise
    
    def _generate_helipath(self) -> np.ndarray:
        """Generate realistic helicopter flight path."""
        # Create waypoints for helicopter
        waypoints = [
            [100, 0, 100],
            [-100, 50, 100],
            [-100, -50, 100],
            [100, 0, 100]
        ]
        return np.array(waypoints)
    
    def close(self):
        """Clean up environment resources."""
        pass