import logging
import os
import sys
from pathlib import Path
from src.config import settings

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(settings.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    file_path = log_path / log_file

    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')

    handler = logging.FileHandler(file_path)        
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)

    return logger
