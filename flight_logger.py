"""
SQL database for logging drone flights and regulatory compliance.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FlightLogger:
    """Database handler for drone flight logging and compliance tracking."""
    
    def __init__(self, db_path: str = "drone_flights.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        
        # Create database directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Flight logger initialized at {db_path}")
    
    def _init_database(self):
        """Initialize database with required tables."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create flights table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS flights (
                flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                weather_condition TEXT,
                helicopter_speed REAL,
                total_steps INTEGER,
                success_rate REAL,
                avg_altitude REAL,
                min_altitude REAL,
                max_altitude REAL,
                collision_count INTEGER,
                safety_violations INTEGER,
                altitude_violations INTEGER,
                total_reward REAL
            )
        """)
        
        # Create flight_steps table (detailed step-by-step data)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS flight_steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id INTEGER,
                step_number INTEGER,
                timestamp TIMESTAMP,
                position_x REAL,
                position_y REAL,
                altitude REAL,
                velocity_x REAL,
                velocity_y REAL,
                distance_to_helicopter REAL,
                action_thrust REAL,
                action_pitch REAL,
                action_roll REAL,
                action_yaw REAL,
                reward REAL,
                safety_margin_violation BOOLEAN,
                altitude_violation BOOLEAN,
                weather_condition TEXT,
                helicopter_speed REAL,
                FOREIGN KEY (flight_id) REFERENCES flights (flight_id)
            )
        """)
        
        # Create violations table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_id INTEGER,
                step_number INTEGER,
                violation_type TEXT,
                violation_value REAL,
                threshold_value REAL,
                timestamp TIMESTAMP,
                position_x REAL,
                position_y REAL,
                altitude REAL,
                description TEXT,
                FOREIGN KEY (flight_id) REFERENCES flights (flight_id)
            )
        """)
        
        # Create indices for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_flights_time 
            ON flights (start_time)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_steps_flight 
            ON flight_steps (flight_id, step_number)
        """)
        
        self.conn.commit()
    
    def start_flight(self, config: Dict) -> int:
        """Start a new flight session and return flight ID."""
        query = """
            INSERT INTO flights 
            (start_time, weather_condition, helicopter_speed)
            VALUES (?, ?, ?)
        """
        
        self.cursor.execute(query, (
            datetime.now(),
            config.get('weather', 'dry'),
            config.get('helicopter_speed', 15.0)
        ))
        
        self.conn.commit()
        flight_id = self.cursor.lastrowid
        
        logger.info(f"Started flight {flight_id}")
        return flight_id
    
    def log_step(self, flight_id: int, step_data: Dict):
        """Log a single timestep of flight data."""
        query = """
            INSERT INTO flight_steps 
            (flight_id, step_number, timestamp, position_x, position_y,
             altitude, velocity_x, velocity_y, distance_to_helicopter,
             action_thrust, action_pitch, action_roll, action_yaw,
             reward, safety_margin_violation, altitude_violation,
             weather_condition, helicopter_speed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Calculate braking performance
        braking_distance, braking_time = self._calculate_braking_metrics(
            step_data.get('velocity', 8.33),
            step_data.get('weather', 'dry')
        )
        
        self.cursor.execute(query, (
            flight_id,
            step_data.get('step_number', 0),
            datetime.now(),
            step_data.get('position', [0, 0, 100])[0],
            step_data.get('position', [0, 0, 100])[1],
            step_data.get('altitude', 100.0),
            step_data.get('velocity', [8.33, 0, 0])[0],
            step_data.get('velocity', [8.33, 0, 0])[1],
            step_data.get('distance_to_helicopter', 50.0),
            step_data.get('action', [0, 0, 0, 0])[0],
            step_data.get('action', [0, 0, 0, 0])[1],
            step_data.get('action', [0, 0, 0, 0])[2],
            step_data.get('action', [0, 0, 0, 0])[3],
            step_data.get('reward', 0.0),
            1 if step_data.get('safety_violation', False) else 0,
            1 if step_data.get('altitude_violation', False) else 0,
            step_data.get('weather', 'dry'),
            step_data.get('helicopter_speed', 15.0)
        ))
        
        # Check for violations
        self._check_and_log_violations(flight_id, step_data)
        
        self.conn.commit()
    
    def _calculate_braking_metrics(self, velocity: float, 
                                   weather: str) -> tuple:
        """Calculate braking distance and time based on weather."""
        # Base deceleration (m/s^2)
        if weather == 'rain':
            deceleration = 0.6 * 9.81  # 40% penalty in rain
        else:
            deceleration = 9.81
        
        braking_distance = (velocity ** 2) / (2 * deceleration)
        braking_time = velocity / deceleration
        
        return braking_distance, braking_time
    
    def _check_and_log_violations(self, flight_id: int, step_data: Dict):
        """Check for regulatory violations and log them."""
        violations = []
        
        # Safety margin violation (15m)
        distance = step_data.get('distance_to_helicopter', 50.0)
        if distance < 15.0:
            violations.append({
                'type': 'safety_margin',
                'value': distance,
                'threshold': 15.0,
                'description': f'Distance to helicopter below safety margin: {distance:.1f}m < 15.0m'
            })
        
        # Altitude violation (120m limit)
        altitude = step_data.get('altitude', 100.0)
        if altitude > 120.0:
            violations.append({
                'type': 'altitude_limit',
                'value': altitude,
                'threshold': 120.0,
                'description': f'Altitude exceeds CAA limit: {altitude:.1f}m > 120.0m'
            })
        
        # Minimum altitude violation (10m)
        if altitude < 10.0:
            violations.append({
                'type': 'minimum_altitude',
                'value': altitude,
                'threshold': 10.0,
                'description': f'Altitude below minimum safe altitude: {altitude:.1f}m < 10.0m'
            })
        
        # Log violations
        for violation in violations:
            query = """
                INSERT INTO violations 
                (flight_id, step_number, violation_type, violation_value,
                 threshold_value, timestamp, position_x, position_y,
                 altitude, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.cursor.execute(query, (
                flight_id,
                step_data.get('step_number', 0),
                violation['type'],
                violation['value'],
                violation['threshold'],
                datetime.now(),
                step_data.get('position', [0, 0, 100])[0],
                step_data.get('position', [0, 0, 100])[1],
                altitude,
                violation['description']
            ))
    
    def end_flight(self, flight_id: int, flight_summary: Dict):
        """Finalize flight session with summary statistics."""
        # Calculate summary statistics
        query = """
            SELECT 
                COUNT(*) as total_steps,
                AVG(reward) as avg_reward,
                AVG(altitude) as avg_altitude,
                MIN(altitude) as min_altitude,
                MAX(altitude) as max_altitude,
                SUM(safety_margin_violation) as safety_violations,
                SUM(altitude_violation) as altitude_violations
            FROM flight_steps
            WHERE flight_id = ?
        """
        
        self.cursor.execute(query, (flight_id,))
        stats = self.cursor.fetchone()
        
        # Update flight record
        update_query = """
            UPDATE flights 
            SET end_time = ?,
                total_steps = ?,
                success_rate = ?,
                avg_altitude = ?,
                min_altitude = ?,
                max_altitude = ?,
                collision_count = ?,
                safety_violations = ?,
                altitude_violations = ?,
                total_reward = ?
            WHERE flight_id = ?
        """
        
        # Calculate success rate (no violations)
        total_steps = stats[0] or 1
        violation_count = (stats[5] or 0) + (stats[6] or 0)
        success_rate = 1.0 - (violation_count / total_steps)
        
        self.cursor.execute(update_query, (
            datetime.now(),
            total_steps,
            success_rate,
            stats[2] or 100.0,
            stats[3] or 100.0,
            stats[4] or 100.0,
            flight_summary.get('collision_count', 0),
            stats[5] or 0,
            stats[6] or 0,
            flight_summary.get('total_reward', 0.0),
            flight_id
        ))
        
        self.conn.commit()
        logger.info(f"Flight {flight_id} ended with success rate: {success_rate:.2f}")
    
    def get_flight_report(self, flight_id: int) -> Dict:
        """Generate comprehensive flight report."""
        # Get flight metadata
        flight_query = """
            SELECT * FROM flights WHERE flight_id = ?
        """
        self.cursor.execute(flight_query, (flight_id,))
        flight_data = self.cursor.fetchone()
        
        if not flight_data:
            return {"error": "Flight not found"}
        
        # Get violations
        violations_query = """
            SELECT * FROM violations WHERE flight_id = ? ORDER BY step_number
        """
        self.cursor.execute(violations_query, (flight_id,))
        violations = self.cursor.fetchall()
        
        # Get flight statistics
        stats_query = """
            SELECT 
                AVG(distance_to_helicopter) as avg_distance,
                STDDEV(distance_to_helicopter) as distance_std,
                AVG(velocity_x) as avg_speed,
                COUNT(DISTINCT step_number) as total_steps
            FROM flight_steps
            WHERE flight_id = ?
        """
        self.cursor.execute(stats_query, (flight_id,))
        stats = self.cursor.fetchone()
        
        # Compile report
        report = {
            "flight_id": flight_id,
            "start_time": flight_data[1],
            "end_time": flight_data[2],
            "weather": flight_data[3],
            "helicopter_speed": flight_data[4],
            "total_steps": flight_data[5],
            "success_rate": flight_data[6],
            "altitude_stats": {
                "average": flight_data[7],
                "minimum": flight_data[8],
                "maximum": flight_data[9]
            },
            "violations": {
                "collisions": flight_data[10],
                "safety_margin": flight_data[11],
                "altitude": flight_data[12]
            },
            "performance_metrics": {
                "average_distance": stats[0] or 50.0,
                "distance_std": stats[1] or 0.0,
                "average_speed": stats[2] or 8.33,
                "total_reward": flight_data[13]
            },
            "violation_details": [
                {
                    "step": v[2],
                    "type": v[3],
                    "value": v[4],
                    "threshold": v[5],
                    "description": v[10]
                } for v in violations
            ]
        }
        
        return report
    
    def export_to_csv(self, flight_id: int, output_dir: str = "reports"):
        """Export flight data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export flight steps
        steps_query = """
            SELECT * FROM flight_steps WHERE flight_id = ? ORDER BY step_number
        """
        steps_df = pd.read_sql_query(steps_query, self.conn, params=(flight_id,))
        steps_df.to_csv(output_path / f"flight_{flight_id}_steps.csv", index=False)
        
        # Export violations
        violations_query = """
            SELECT * FROM violations WHERE flight_id = ? ORDER BY step_number
        """
        violations_df = pd.read_sql_query(violations_query, self.conn, params=(flight_id,))
        violations_df.to_csv(output_path / f"flight_{flight_id}_violations.csv", index=False)
        
        logger.info(f"Flight {flight_id} data exported to {output_dir}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")