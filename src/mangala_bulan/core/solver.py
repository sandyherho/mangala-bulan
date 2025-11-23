import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm
import os
from numba import njit, prange, set_num_threads
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
            print(f"  Grid: {nx} points, dx = {self.dx:.6e} cm")
            print(f"  Domain: [0, {L:.6e}] cm")
            print(f"  CPU cores: {self.n_cores}")
    
    def solve(self, config: Dict[str, Any]):
        method = config.get('method', 'ftcs')
        
        if method == 'ftcs':
            result = self._solve_ftcs(config)
        elif method == 'dufort':
            result = self._solve_dufort_frankel(config)
        elif method == 'crank':
            result = self._solve_crank_nicolson(config)
        elif method == 'laasonen':
            result = self._solve_laasonen(config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save outputs
        scenario = config.get('scenario_name', 'simulation')
        clean_name = scenario.replace(' ', '_').replace('-', '_').lower()
        
        if config.get('save_netcdf', True):
            DataHandler.save_netcdf(f"{clean_name}.nc", result, config)
        
        if config.get('save_csv', True):
            DataHandler.save_csv(f"{clean_name}.csv", result, config)
        
        if config.get('save_animation', True):
            Animator.create_gif(result, f"{clean_name}.gif", config)
        
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
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"  Method: FTCS (Explicit)")
            print(f"  dt = {dt:.6e} s, Stability = {d:.4f}")
            pbar = tqdm(total=nt-1, desc="  Integrating", unit="step",
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]')
        
        for n in range(nt-1):
            T_mb = compute_myoglobin_equilibrium(T[n], P50)
            MM_term = compute_michaelis_menten(T[n], V_max, K_m)
            
            for j in range(1, self.nx-1):
                diffusion = d * (T[n, j+1] - 2*T[n, j] + T[n, j-1])
                mb_diffusion = d_mb * (T_mb[j+1] - 2*T_mb[j] + T_mb[j-1])
                T[n+1, j] = T[n, j] + diffusion - dt*MM_term[j] + mb_diffusion
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
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
            print(f"  Method: DuFort-Frankel (Explicit)")
            print(f"  dt = {dt:.6e} s")
            pbar = tqdm(total=nt-2, desc="  Integrating", unit="step",
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
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"  Method: Crank-Nicolson (Implicit)")
            print(f"  dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc="  Integrating", unit="step",
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
        
        nt = int(ts / dt) + 1
        t = np.linspace(0, ts, nt)
        
        T = np.zeros((nt, self.nx))
        T[:, 0] = T0
        T[:, -1] = Ts
        T[0, 1:-1] = 0
        
        if self.verbose:
            print(f"  Method: Laasonen (Implicit)")
            print(f"  dt = {dt:.6e} s")
            pbar = tqdm(total=nt-1, desc="  Integrating", unit="step",
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
