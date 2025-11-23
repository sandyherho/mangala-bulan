import logging
from pathlib import Path
from datetime import datetime

class SimulationLogger:
    def __init__(self, scenario_name: str, log_dir: str = "logs", verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.warnings = []
        self.errors = []
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger(f"mangala_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        self.warnings.append(msg)
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        self.logger.error(msg)
        self.errors.append(msg)
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def finalize(self):
        self.info("="*60)
        self.info("SIMULATION SUMMARY")
        self.info("="*60)
        self.info(f"Scenario: {self.scenario_name}")
        self.info(f"Warnings: {len(self.warnings)}")
        self.info(f"Errors: {len(self.errors)}")
        self.info(f"Log file: {self.log_file}")
        self.info("="*60)
