"""
Custom logger for training and evaluation metrics.
"""

import logging
from datetime import datetime
from pathlib import Path
import json

class TrainingLogger:
    """Logger for tracking training metrics and saving results."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file