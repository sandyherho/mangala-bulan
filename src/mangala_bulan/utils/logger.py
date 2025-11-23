import logging
from pathlib import Path
from datetime import datetime
import time

class SimulationLogger:
    def __init__(self, scenario_name: str, log_dir: str = "logs", verbose: bool = True):
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.warnings = []
        self.errors = []
        self.timing = {}
        self.start_time = None
        self.phase_start_time = None
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Remove date from filename - just use scenario name
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.start_simulation()
        
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
    
    def start_simulation(self):
        """Record simulation start time"""
        self.start_time = time.time()
        self.info("="*80)
        self.info(f"SIMULATION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("="*80)
        self.info(f"Scenario: {self.scenario_name}")
        self.info(f"Process ID: {os.getpid() if 'os' in globals() else 'N/A'}")
        self.info(f"Python version: {sys.version if 'sys' in globals() else 'N/A'}")
        self.info("="*80)
    
    def start_phase(self, phase_name: str):
        """Start timing a specific phase"""
        self.phase_start_time = time.time()
        self.info(f"Starting phase: {phase_name}")
        
    def end_phase(self, phase_name: str, details: dict = None):
        """End timing a specific phase and log details"""
        if self.phase_start_time:
            elapsed = time.time() - self.phase_start_time
            self.timing[phase_name] = elapsed
            self.info(f"Completed phase: {phase_name} - Time: {elapsed:.6f} seconds")
            if details:
                for key, value in details.items():
                    self.info(f"  {key}: {value}")
            self.phase_start_time = None
    
    def log_physics_params(self, params: dict):
        """Log physical parameters"""
        self.info("="*80)
        self.info("PHYSICAL PARAMETERS")
        self.info("="*80)
        for key, value in params.items():
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e3:
                    self.info(f"{key:25s}: {value:15.6e}")
                else:
                    self.info(f"{key:25s}: {value:15.6f}")
            else:
                self.info(f"{key:25s}: {value}")
        self.info("="*80)
    
    def log_numerical_params(self, params: dict):
        """Log numerical parameters"""
        self.info("="*80)
        self.info("NUMERICAL PARAMETERS")
        self.info("="*80)
        for key, value in params.items():
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e3:
                    self.info(f"{key:25s}: {value:15.6e}")
                else:
                    self.info(f"{key:25s}: {value:15.6f}")
            else:
                self.info(f"{key:25s}: {value}")
        self.info("="*80)
    
    def log_stability_analysis(self, stability: dict):
        """Log stability analysis"""
        self.info("="*80)
        self.info("STABILITY ANALYSIS")
        self.info("="*80)
        for key, value in stability.items():
            if isinstance(value, float):
                self.info(f"{key:25s}: {value:15.6f}")
            else:
                self.info(f"{key:25s}: {value}")
        self.info("="*80)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        self.warnings.append(msg)
        if self.verbose:
            print(f"{'':>20}WARNING: {msg}")
    
    def error(self, msg: str):
        self.logger.error(msg)
        self.errors.append(msg)
        if self.verbose:
            print(f"{'':>20}ERROR: {msg}")
    
    def finalize(self):
        """Finalize logging with complete timing summary"""
        if self.start_time:
            total_time = time.time() - self.start_time
            
            self.info("="*80)
            self.info("COMPUTATIONAL TIMING SUMMARY")
            self.info("="*80)
            
            # Phase timings
            if self.timing:
                for phase, duration in self.timing.items():
                    percentage = (duration / total_time) * 100 if total_time > 0 else 0
                    self.info(f"{phase:30s}: {duration:10.6f} s ({percentage:5.2f}%)")
                self.info("-"*80)
            
            # Total time
            self.info(f"{'Total computation time':30s}: {total_time:10.6f} s")
            self.info(f"{'Wall clock time':30s}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Performance metrics
            if 'integration' in self.timing:
                integration_time = self.timing['integration']
                self.info(f"{'Integration rate':30s}: {1/integration_time if integration_time > 0 else 0:10.2f} steps/s")
            
            self.info("="*80)
            self.info("SIMULATION SUMMARY")
            self.info("="*80)
            self.info(f"Scenario: {self.scenario_name}")
            self.info(f"Status: {'COMPLETED' if len(self.errors) == 0 else 'FAILED'}")
            self.info(f"Warnings: {len(self.warnings)}")
            self.info(f"Errors: {len(self.errors)}")
            self.info(f"Log file: {self.log_file}")
            self.info(f"Total runtime: {total_time:.6f} seconds")
            self.info("="*80)
            self.info(f"SIMULATION ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.info("="*80)

# Import these at module level to avoid issues
import os
import sys
