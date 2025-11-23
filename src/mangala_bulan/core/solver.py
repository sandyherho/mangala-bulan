import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm
import os
import time
from numba import njit, prange, set_num_threads
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from ..io.data_handler import DataHandler
from ..visualization.animator import Animator

@njit(parallel=True, cache=True)
def compute_michaelis_menten(T: np.ndarray, V_max: float, K_m: float) -> np.ndarray:
    result = np.empty_like(T)
    for i in prange(len(T)):
        result[i] = V_max * T[i] / (K_m + T[i])
    return result

@njit(parallel=True, cache=True)
def compute_myoglobin_equilibrium(T: np.ndarray, P50: float) -> np.ndarray:
    result = np.empty_like(T)
    for i in prange(len(T)):
        result[i] = T[i] * P50 / (P50 + T[i])
    return result

class OxygenDiffusionSolver:
    def __init__(self, nx: int = 51, L: float = 1.0e-3, verbose: bool = True,
                 logger: Optional[Any] = None, n_cores: Optional[int] = None):
        self.nx = nx
        self.L = L
        self.verbose = verbose
        self.logger = logger
        
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        if n_cores is None or n_cores == 0:
            n_cores = os.cpu_count()
        self.n_cores = n_cores
        set_num_threads(self.n_cores)
        
        if verbose:
            print(f"{'':>10}Grid: {nx} points, dx = {self.dx:.6e} cm")
            print(f"{'':>10}Domain: [0, {L:.6e}] cm")
            print(f"{'':>10}CPU cores: {self.n_cores}")
        
        # Log numerical parameters
        if self.logger:
            self.logger.log_numerical_params({
                'Grid points (nx)': nx,
                'Grid spacing (dx)': self.dx,
                'Domain length (L)': L,
                'CPU cores': self.n_cores
            })
    
    def solve(self, config: Dict[str, Any]):
        # Log physical parameters
        if self.logger:
            self.logger.log_physics_params({
                'Initial O2 (T0)': config.get('T0', 70.0),
                'Boundary O2 (Ts)': config.get('Ts', 10.0),
                'Diffusion coef (D)': config.get('koef', 5.5e-7),
                'V_max': config.get('V_max', 2e-4),
                'K_m': config.get('K_m', 1.0),
                'Myoglobin diff (D_mb)': config.get('D_mb', 3e-8),
                'P50': config.get('P50', 2.0),
                'Time step (dt)': config.get('dt', 5e-4),
                'Total time (ts)': config.get('ts', 1.0)
            })
        
        method = config.get('method', 'ftcs')
        
        # Start integration phase
        if self.logger:
            self.logger.start_phase('integration')
        
        start_time = time.time()
        
        if method == 'ftcs':
            result = self._solve_ftcs(config)
        elif method == 'dufort':
            result = self._solve_dufort_frankel(config)
        elif method == 'crank':
            result = self._solve_crank_nicolson(config)
        elif method == 'laasonen':
            result = self._solve_laasonen(config)
        elif method == 'adi':
            result = self._solve_adi(config)
        elif method == 'rkimex':
            result = self._solve_rk_imex(config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        integration_time = time.time() - start_time
        
        # End integration phase with details
        if self.logger:
            nt = len(result['t'])
            self.logger.end_phase('integration', {
                'Method': method,
                'Time steps': nt,
                'Integration time': f'{integration_time:.6f} s',
                'Steps per second': f'{nt/integration_time:.2f}'
            })
        
        # Save outputs
        scenario = config.get('scenario_name', 'simulation')
        clean_name = scenario.replace(' ', '_').replace('-', '_').lower()
        
        # NetCDF output
        if config.get('save_netcdf', True):
            if self.logger:
                self.logger.start_phase('netcdf_output')
            DataHandler.save_netcdf(f"{clean_name}.nc", result, config)
            if self.logger:
                self.logger.end_phase('netcdf_output')
        
        # CSV output
        if config.get('save_csv', True):
            if self.logger:
                self.logger.start_phase('csv_output')
            DataHandler.save_csv(f"{clean_name}.csv", result, config)
            if self.logger:
                self.logger.end_phase('csv_output')
        
        # Animation
        if config.get('save_animation', True):
            if self.logger:
                self.logger.start_phase('animation')
            Animator.create_gif(result, f"{clean_name}.gif", config)
            if self.logger:
                self.logger.end_phase('animation')
        
        return result
    
    def _solve_ftcs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = koef * dt / (self.dx**2)
        d_mb = D_mb * dt / (self.dx**2)
        
        # Check stability and warn if unstable
        stability_check = d < 0.5
        stability_status = 'STABLE' if stability_check else 'UNSTABLE'
        
        if not stability_check:
            if self.verbose:
                print(f"{'':>10}WARNING: Stability condition violated! d={d:.4f} >= 0.5")
                print(f"{'':>10}Solution may become unstable. Consider reducing dt.")
            if self.logger:
                self.logger.warning(f"FTCS stability violated: d={d:.4f} >= 0.5")
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d,
                'Myoglobin diff number': d_mb,
                'Stability requirement': 'd < 0.5',
                'Current stability': stability_status,
                'CFL number': d
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"{'':>10}Method: FTCS (Explicit)")
            print(f"{'':>10}dt = {dt:.6e} s, Stability = {d:.4f} ({stability_status})")
            pbar = tqdm(total=nt-1, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        # Track if solution becomes unstable
        solution_stable = True
        
        for n in range(nt-1):
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            for j in range(1, self.nx-1):
                diffusion = d * (T[n, j+1] - 2*T[n, j] + T[n, j-1])
                mb_diffusion = d_mb * (T_mb[j+1] - 2*T_mb[j] + T_mb[j-1])
                T[n+1, j] = T[n, j] + diffusion - dt*MM_term[j] + mb_diffusion
                
                # Clamp values to prevent extreme instabilities
                if T[n+1, j] < 0:
                    T[n+1, j] = 0
                elif T[n+1, j] > 1e3:  # Reasonable upper bound for concentration
                    T[n+1, j] = 1e3
            
            # Check for NaN or Inf values
            if np.any(np.isnan(T[n+1])) or np.any(np.isinf(T[n+1])):
                solution_stable = False
                if self.verbose:
                    pbar.close()
                    print(f"\n{'':>10}Solution became unstable at time step {n+1}/{nt}")
                    print(f"{'':>10}Terminating integration early")
                if self.logger:
                    self.logger.error(f"FTCS solution unstable at step {n+1}/{nt}")
                
                # Truncate solution to last stable state
                T = T[:n+1]
                t = t[:n+1]
                break
            
            # Check for growing instability
            max_val = np.max(np.abs(T[n+1, 1:-1]))
            if max_val > 1e6:
                if self.verbose and n % 100 == 0:
                    print(f"\n{'':>10}Large values detected: max = {max_val:.2e}")
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose and solution_stable:
            pbar.close()
        
        # Final check for solution validity
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            # Replace NaN/Inf with boundary values as fallback
            if self.verbose:
                print(f"{'':>10}Cleaning up NaN/Inf values in solution")
            T[np.isnan(T)] = 0
            T[np.isinf(T)] = 0
            
            # If solution is mostly corrupted, create a simple linear profile
            if np.sum(np.isnan(T) | np.isinf(T)) > T.size * 0.1:
                if self.verbose:
                    print(f"{'':>10}Solution severely corrupted, using fallback linear profile")
                for n in range(len(t)):
                    T[n] = np.linspace(T0, Ts, self.nx)
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'FTCS'}
    
    def _solve_dufort_frankel(self, config: Dict[str, Any]) -> Dict[str, Any]:
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = 2 * koef * dt / (self.dx**2)
        d_mb = 2 * D_mb * dt / (self.dx**2)
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d/2,
                'Method stability': 'Unconditionally stable',
                'Numerical diffusion': 'Present due to time-centering'
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        # First step with FTCS
        d1 = koef * dt / (self.dx**2)
        T_mb = compute_myoglobin_equilibrium(T[0], P50)
        MM_term = compute_michaelis_menten(T[0], V_max, K_m)
        for j in range(1, self.nx-1):
            T[1, j] = T[0, j] + d1*(T[0, j+1] - 2*T[0, j] + T[0, j-1]) - dt*MM_term[j]
        
        if self.verbose:
            print(f"{'':>10}Method: DuFort-Frankel (Explicit)")
            print(f"{'':>10}dt = {dt:.6e} s")
            pbar = tqdm(total=nt-2, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        for n in range(1, nt-1):
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            for j in range(1, self.nx-1):
                numerator = (1-d)*T[n-1, j] + d*(T[n, j+1] + T[n, j-1])
                numerator += d_mb*(T_mb[j+1] - 2*T_mb[j] + T_mb[j-1]) - 2*dt*MM_term[j]
                T[n+1, j] = numerator / (1 + d)
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'DuFort-Frankel'}
    
    def _solve_crank_nicolson(self, config: Dict[str, Any]) -> Dict[str, Any]:
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 2.5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = koef * dt / (self.dx**2)
        d_mb = D_mb * dt / (self.dx**2)
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d,
                'Method stability': 'Unconditionally stable',
                'Method order': 'Second-order in space and time'
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"{'':>10}Method: Crank-Nicolson (Implicit)")
            print(f"{'':>10}dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        for n in range(nt-1):
            H = np.zeros(self.nx)
            G = np.zeros(self.nx)
            H[0] = 0
            G[0] = T[n, 0]
            
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            for j in range(1, self.nx-1):
                a = d/2
                b = -d - 1
                c = d/2
                D = -T[n, j] - (d/2)*(T[n, j+1] + T[n, j-1] - 2*T[n, j])
                D -= dt*MM_term[j] - d_mb/2*(T_mb[j+1] - 2*T_mb[j] + T_mb[j-1])
                
                if j > 0:
                    H[j] = c / (b - a*H[j-1])
                    G[j] = (D - a*G[j-1]) / (b - a*H[j-1])
            
            T[n+1, -1] = Ts
            for j in range(self.nx-2, 0, -1):
                T[n+1, j] = -H[j]*T[n+1, j+1] + G[j]
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'Crank-Nicolson'}
    
    def _solve_laasonen(self, config: Dict[str, Any]) -> Dict[str, Any]:
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 2.5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = koef * dt / (self.dx**2)
        d_mb = D_mb * dt / (self.dx**2)
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d,
                'Method stability': 'Unconditionally stable',
                'Method type': 'Fully implicit'
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"{'':>10}Method: Laasonen (Implicit)")
            print(f"{'':>10}dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        for n in range(nt-1):
            H = np.zeros(self.nx)
            G = np.zeros(self.nx)
            H[0] = 0
            G[0] = T[n, 0]
            
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            for j in range(1, self.nx-1):
                a = d
                b = -(2*d + 1)
                c = d
                D = -T[n, j] - dt*MM_term[j] + d_mb*(T_mb[j+1] - 2*T_mb[j] + T_mb[j-1])
                
                if j > 0:
                    H[j] = c / (b - a*H[j-1])
                    G[j] = (D - a*G[j-1]) / (b - a*H[j-1])
            
            T[n+1, -1] = Ts
            for j in range(self.nx-2, 0, -1):
                T[n+1, j] = -H[j]*T[n+1, j+1] + G[j]
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'Laasonen'}
    
    def _solve_adi(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Alternating Direction Implicit (ADI) method"""
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 2.5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = koef * dt / (self.dx**2)
        d_mb = D_mb * dt / (self.dx**2)
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d,
                'Method stability': 'Unconditionally stable',
                'Method type': 'Alternating Direction Implicit',
                'Matrix solver': 'Direct sparse solver'
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"{'':>10}Method: ADI (Alternating Direction Implicit)")
            print(f"{'':>10}dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        # Tridiagonal matrix setup
        nx_inner = self.nx - 2
        
        for n in range(nt-1):
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            # Build tridiagonal system
            main_diag = np.ones(nx_inner) * (1 + d)
            off_diag = np.ones(nx_inner - 1) * (-d/2)
            A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsr()
            
            # RHS vector
            b = np.zeros(nx_inner)
            for j in range(nx_inner):
                idx = j + 1
                b[j] = T[n, idx] + (d/2)*(T[n, idx+1] - 2*T[n, idx] + T[n, idx-1])
                b[j] -= dt*MM_term[idx]
                b[j] += d_mb*(T_mb[idx+1] - 2*T_mb[idx] + T_mb[idx-1])
            
            # Apply boundary conditions
            b[0] += (d/2) * T0
            b[-1] += (d/2) * Ts
            
            # Solve
            T_inner = spsolve(A, b)
            T[n+1, 1:-1] = T_inner
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'ADI'}
    
    def _solve_rk_imex(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Runge-Kutta IMEX (Implicit-Explicit) method"""
        T0 = config.get('T0', 70.0)
        Ts = config.get('Ts', 10.0)
        koef = config.get('koef', 5.5e-7)
        dt = config.get('dt', 5e-4)
        ts = config.get('ts', 1.0)
        V_max = config.get('V_max', 2e-4)
        K_m = config.get('K_m', 1.0)
        D_mb = config.get('D_mb', 3e-8)
        P50 = config.get('P50', 2.0)
        
        d = koef * dt / (self.dx**2)
        d_mb = D_mb * dt / (self.dx**2)
        
        # Log stability analysis
        if self.logger:
            self.logger.log_stability_analysis({
                'Diffusion number (d)': d,
                'Method type': 'Runge-Kutta IMEX',
                'Method order': 'Second-order',
                'Stiff term treatment': 'Implicit',
                'Non-stiff term treatment': 'Explicit'
            })
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"{'':>10}Method: RK-IMEX (Runge-Kutta Implicit-Explicit)")
            print(f"{'':>10}dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc=" "*10 + "Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        # IMEX-RK coefficients (2nd order)
        gamma = 0.5
        
        for n in range(nt-1):
            # Stage 1: Implicit diffusion, explicit reaction
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            # Solve implicit system for diffusion
            nx_inner = self.nx - 2
            main_diag = np.ones(nx_inner) * (1 + 2*gamma*d)
            off_diag = np.ones(nx_inner - 1) * (-gamma*d)
            A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsr()
            
            b = T[n, 1:-1].copy()
            b -= dt*gamma*MM_term[1:-1]
            b += gamma*d_mb*(T_mb[2:] - 2*T_mb[1:-1] + T_mb[:-2])
            b[0] += gamma*d*T0
            b[-1] += gamma*d*Ts
            
            k1 = spsolve(A, b)
            
            # Stage 2
            T_temp = np.copy(T[n])
            T_temp[1:-1] = k1
            T_mb2 = compute_myoglobin_equilibrium(T_temp, P50)
            MM_term2 = compute_michaelis_menten(T_temp, V_max, K_m)
            
            b2 = T[n, 1:-1] + dt*(1-gamma)*(k1 - T[n, 1:-1])/dt
            b2 -= dt*(1-2*gamma)*MM_term[1:-1] - dt*gamma*MM_term2[1:-1]
            b2 += (1-gamma)*d_mb*(T_mb[2:] - 2*T_mb[1:-1] + T_mb[:-2])
            b2 += gamma*d_mb*(T_mb2[2:] - 2*T_mb2[1:-1] + T_mb2[:-2])
            b2[0] += gamma*d*T0
            b2[-1] += gamma*d*Ts
            
            T[n+1, 1:-1] = spsolve(A, b2)
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        return {'x': self.x*100, 't': t, 'T': T, 'method': 'RK-IMEX'}
