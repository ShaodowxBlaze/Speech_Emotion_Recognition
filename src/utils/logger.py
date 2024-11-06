import logging
from pathlib import Path

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()
        ]
    )