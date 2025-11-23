import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime

class DataHandler:
    @staticmethod
    def save_netcdf(filename: str, result: dict, config: dict, output_dir: str = "outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            nx = len(result['x'])
            nt = len(result['t'])
            nc.createDimension('x', nx)
            nc.createDimension('t', nt)
            
            nc_x = nc.createVariable('x', 'f4', ('x',), zlib=True)
            nc_x[:] = result['x']
            nc_x.units = "mm"
            nc_x.long_name = "position"
            
            nc_t = nc.createVariable('t', 'f4', ('t',), zlib=True)
            nc_t[:] = result['t']
            nc_t.units = "s"
            nc_t.long_name = "time"
            
            nc_T = nc.createVariable('oxygen_concentration', 'f4', ('t', 'x'), zlib=True)
            nc_T[:] = result['T']
            nc_T.units = "mg/ml"
            nc_T.long_name = "oxygen_concentration"
            
            nc.method = result['method']
            nc.T0 = float(config.get('T0', 70.0))
            nc.T0_units = "mg/ml"
            nc.Ts = float(config.get('Ts', 10.0))
            nc.Ts_units = "mg/ml"
            nc.diffusion_coefficient = float(config.get('koef', 5.5e-7))
            nc.diffusion_coefficient_units = "cm2/s"
            nc.V_max = float(config.get('V_max', 2e-4))
            nc.V_max_units = "mg/ml/s"
            nc.K_m = float(config.get('K_m', 1.0))
            nc.K_m_units = "mg/ml"
            nc.myoglobin_diffusion = float(config.get('D_mb', 3e-8))
            nc.myoglobin_diffusion_units = "cm2/s"
            nc.P50 = float(config.get('P50', 2.0))
            nc.P50_units = "mg/ml"
            
            nc.created = datetime.now().isoformat()
            nc.software = "mangala-bulan"
            nc.version = "0.0.2"
            nc.title = f"Oxygen Diffusion: {config.get('scenario_name', 'unknown')}"
    
    @staticmethod
    def save_csv(filename: str, result: dict, config: dict, output_dir: str = "outputs"):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save final state
        df = pd.DataFrame({
            'x_mm': result['x'],
            'initial': result['T'][0],
            'final': result['T'][-1]
        })
        df.to_csv(output_path / filename, index=False)
        
        # Save time series at key points
        key_points = [5, 15, 25, 35, 45] if len(result['x']) > 45 else [len(result['x'])//2]
        ts_data = {}
        ts_data['time_s'] = result['t']
        for idx in key_points:
            if idx < len(result['x']):
                ts_data[f'grid_{idx}'] = result['T'][:, idx]
        
        df_ts = pd.DataFrame(ts_data)
        ts_filename = filename.replace('.csv', '_timeseries.csv')
        df_ts.to_csv(output_path / ts_filename, index=False)
