#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from .core.solver import OxygenDiffusionSolver
from .io.config_manager import ConfigManager
from .utils.logger import SimulationLogger

def print_header():
    print("\n" + "="*70)
    print(" "*15 + "mangala-bulan 1D Idealized Oxygen Diffusion Solver")
    print(" "*20 + "Version 0.0.1")
    print("="*70)
    print("\n  Oxygen Diffusion with Michaelis-Menten & Myoglobin")
    print("  Authors: Sandy H. S. Herho, Gandhi Napitupulu")
    print("="*70 + "\n")

def run_scenario(config_path: str, verbose: bool = True):
    config = ConfigManager.load(config_path)
    scenario = config.get('scenario_name', 'simulation')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*60}")
    
    logger = SimulationLogger(scenario.replace(' ', '_'), "logs", verbose)
    
    try:
        solver = OxygenDiffusionSolver(
            nx=config.get('nx', 51),
            L=config.get('L', 1.0e-3),
            verbose=verbose,
            logger=logger,
            n_cores=config.get('n_cores', 0)
        )
        
        solver.solve(config)
        
        if verbose:
            print(f"\n{'='*60}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"{'='*60}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        if verbose:
            print(f"\nSIMULATION FAILED: {str(e)}\n")
        raise
    finally:
        logger.finalize()

def main():
    parser = argparse.ArgumentParser(description='mangala-bulan 1D Idealized Oxygen Diffusion Solver')
    parser.add_argument('config', nargs='?', help='Config file path')
    parser.add_argument('--all', '-a', action='store_true', help='Run all configs')
    parser.add_argument('--method', '-m', choices=['ftcs','dufort','crank','laasonen'])
    parser.add_argument('--quiet', '-q', action='store_true')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.all:
        configs = sorted(Path('configs').glob('*.txt'))
        for cfg in configs:
            run_scenario(str(cfg), verbose)
    elif args.method:
        configs = sorted(Path('configs').glob(f'{args.method}_*.txt'))
        for cfg in configs:
            run_scenario(str(cfg), verbose)
    elif args.config:
        run_scenario(args.config, verbose)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
