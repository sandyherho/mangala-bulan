# `mangala-bulan`: 1D Idealized Oxygen Diffusion Solver

[![PyPI version](https://badge.fury.io/py/mangala-bulan.svg)](https://badge.fury.io/py/mangala-bulan)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mangala-bulan.svg)](https://pypi.org/project/mangala-bulan/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.20-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-%E2%89%A51.7-8CAAE6.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%E2%89%A53.3-11557c.svg)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%E2%89%A51.3-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Numba](https://img.shields.io/badge/numba-%E2%89%A50.53-00A3E0.svg)](https://numba.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%E2%89%A51.5-008080.svg)](https://unidata.github.io/netcdf4-python/)
[![tqdm](https://img.shields.io/badge/tqdm-%E2%89%A54.60-FFC107.svg)](https://tqdm.github.io/)
[![imageio](https://img.shields.io/badge/imageio-%E2%89%A52.9-009688.svg)](https://imageio.github.io/)

A high-performance numerical solver for 1D idealized oxygen diffusion in biological tissues with Michaelis-Menten oxygen consumption kinetics and myoglobin-facilitated diffusion.

## Key Features

- **Six numerical methods**: FTCS, DuFort-Frankel, Crank-Nicolson, Laasonen, ADI, and RK-IMEX
- **High performance**: Numba JIT compilation with automatic parallelization
- **Flexible output formats**: NetCDF4, CSV, and animated visualizations
- **Stability monitoring**: Real-time detection and handling of numerical instabilities
- **Comprehensive logging**: Detailed computational performance metrics

## Requirements

### System Requirements
- Operating System: Windows, macOS, or Linux
- RAM: 4 GB minimum (8 GB recommended for large simulations)
- Disk Space: 500 MB for installation + space for outputs

### Python Dependencies
| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | ≥1.20.0 | Numerical arrays and operations |
| scipy | ≥1.7.0 | Sparse matrix solvers |
| matplotlib | ≥3.3.0 | Visualization and plotting |
| netCDF4 | ≥1.5.0 | Scientific data storage |
| pandas | ≥1.3.0 | Data analysis and CSV export |
| tqdm | ≥4.60.0 | Progress bars |
| numba | ≥0.53.0 | JIT compilation and parallelization |
| imageio | ≥2.9.0 | GIF animation creation |

## Governing Equations

The solver implements the coupled reaction-diffusion system:

$$\frac{\partial C_{O_2}}{\partial t} = D_{O_2} \frac{\partial^2 C_{O_2}}{\partial x^2} - \frac{V_{max} C_{O_2}}{K_m + C_{O_2}} + D_{Mb} \frac{\partial^2 C_{Mb}}{\partial x^2}$$

where myoglobin concentration follows Hill equilibrium:

$$C_{Mb} = \frac{C_{O_2} \cdot P_{50}}{P_{50} + C_{O_2}}$$

### Physical Parameters

| Symbol | Description | Default Value | Unit |
|--------|-------------|---------------|------|
| $C_{O_2}$ | Oxygen concentration | - | mg/ml |
| $C_{Mb}$ | Myoglobin-bound oxygen | - | mg/ml |
| $D_{O_2}$ | Oxygen diffusion coefficient | $5.5 \times 10^{-7}$ | cm²/s |
| $D_{Mb}$ | Myoglobin diffusion coefficient | $3.0 \times 10^{-8}$ | cm²/s |
| $V_{max}$ | Maximum oxygen consumption rate | $2.0 \times 10^{-4}$ | mg/(ml·s) |
| $K_m$ | Michaelis constant | 1.0 | mg/ml |
| $P_{50}$ | Half-saturation pressure | 2.0 | mg/ml |
| $L$ | Tissue domain length | $1.0 \times 10^{-3}$ | cm |

### Boundary Conditions

- **Left boundary (x = 0)**: $C_{O_2}(0, t) = T_0$ (arterial oxygen level)
- **Right boundary (x = L)**: $C_{O_2}(L, t) = T_s$ (venous oxygen level)
- **Initial condition**: $C_{O_2}(x, 0) = 0$ for $0 < x < L$

## Numerical Methods

The solver implements six finite difference schemes:

### 1. **FTCS** (Forward-Time Central-Space)
- Explicit scheme: $C_i^{n+1} = C_i^n + d(C_{i+1}^n - 2C_i^n + C_{i-1}^n) - \Delta t \cdot R_i^n$
- Stability: Requires $d = \frac{D_{O_2} \Delta t}{\Delta x^2} < 0.5$
- Order: $O(\Delta t, \Delta x^2)$

### 2. **DuFort-Frankel**
- Explicit three-level scheme: $\frac{C_i^{n+1} - C_i^{n-1}}{2\Delta t} = D_{O_2} \frac{C_{i+1}^n - (C_i^{n+1} + C_i^{n-1}) + C_{i-1}^n}{\Delta x^2}$
- Stability: Unconditionally stable
- Order: $O(\Delta t^2, \Delta x^2, (\Delta t/\Delta x)^2)$

### 3. **Crank-Nicolson**
- Semi-implicit scheme: $\frac{C_i^{n+1} - C_i^n}{\Delta t} = \frac{D_{O_2}}{2} \left[\delta_x^2 C_i^{n+1} + \delta_x^2 C_i^n\right]$
- Stability: Unconditionally stable
- Order: $O(\Delta t^2, \Delta x^2)$

### 4. **Laasonen** (Fully Implicit)
- Implicit scheme: $C_i^{n+1} = C_i^n + d(C_{i+1}^{n+1} - 2C_i^{n+1} + C_{i-1}^{n+1})$
- Stability: Unconditionally stable
- Order: $O(\Delta t, \Delta x^2)$

### 5. **ADI** (Alternating Direction Implicit)
- Tridiagonal matrix solver with sparse LU decomposition
- Stability: Unconditionally stable
- Order: $O(\Delta t^2, \Delta x^2)$

### 6. **RK-IMEX** (Runge-Kutta Implicit-Explicit)
- Second-order IMEX scheme treating stiff diffusion implicitly
- Stability: L-stable for stiff terms
- Order: $O(\Delta t^2, \Delta x^2)$

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from PyPI

The simplest way to install `mangala-bulan` is via pip:

```bash
pip install mangala-bulan
```

### Development Installation

For development or to get the latest unreleased features:

```bash
git clone https://github.com/sandyherho/mangala-bulan.git
cd mangala-bulan
pip install -e .
```

### Install with Poetry (for contributors)

```bash
git clone https://github.com/username/mangala-bulan.git
cd mangala-bulan
poetry install --with dev
```

### Verify Installation

```python
import mangala_bulan
print(mangala_bulan.__version__)
# Output: 0.0.1
```

## Quick Start

### 1. Basic Usage

```python
from mangala_bulan import OxygenDiffusionSolver

# Create solver with default parameters
solver = OxygenDiffusionSolver(nx=51, L=1.0e-3)

# Define configuration
config = {
    'method': 'crank',  # Crank-Nicolson method
    'T0': 70.0,        # Initial O2 concentration [mg/ml]
    'Ts': 10.0,        # Boundary O2 concentration [mg/ml]
    'koef': 5.5e-7,    # Diffusion coefficient [cm²/s]
    'dt': 1.0/300,     # Time step [s]
    'ts': 1.0,         # Total simulation time [s]
    'V_max': 2.0e-4,   # Max consumption rate [mg/(ml·s)]
    'K_m': 1.0,        # Michaelis constant [mg/ml]
    'D_mb': 3.0e-8,    # Myoglobin diffusion [cm²/s]
    'P50': 2.0,        # Half-saturation [mg/ml]
    'save_netcdf': True,
    'save_animation': True
}

# Run simulation
result = solver.solve(config)

# Access results
x_positions = result['x']  # Spatial coordinates [mm]
time_points = result['t']   # Time points [s]
oxygen_conc = result['T']   # O2 concentration [mg/ml]
```

### 2. Command Line Interface

Run a single scenario:
```bash
mangala-bulan configs/crank_scenario1.txt
```


## Configuration Format

Configuration files use simple key-value pairs:
```ini
scenario_name = FTCS Method - Scenario 1
method = ftcs
T0 = 70.0          # Initial O2 concentration [mg/ml]
Ts = 10.0          # Boundary O2 concentration [mg/ml]
koef = 5.5e-7      # Diffusion coefficient [cm²/s]
L = 1.0e-3         # Domain length [cm]
ts = 1.0           # Simulation time [s]
dt = 1.0/2000      # Time step [s]
nx = 51            # Grid points
V_max = 2.0e-4     # Max consumption rate [mg/(ml·s)]
K_m = 1.0          # Michaelis constant [mg/ml]
D_mb = 3.0e-8      # Myoglobin diffusion [cm²/s]
P50 = 2.0          # Half-saturation [mg/ml]
```

## Output Formats

### NetCDF4 Files
Complete solution data with metadata:
- Dimensions: `time × space`
- Variables: `oxygen_concentration(t,x)`, `x(x)`, `t(t)`
- Attributes: All physical and numerical parameters

### Animations
- GIF animations showing spatiotemporal evolution
- Customizable colormap, frame rate, and resolution

## Performance Features

- **Parallel Computing**: Automatic CPU core detection with Numba JIT compilation
- **Sparse Matrix Solvers**: Efficient tridiagonal systems for implicit methods
- **Memory Optimization**: Streaming I/O for large datasets
- **Stability Monitoring**: Real-time detection and handling of numerical instabilities

## Numerical Stability

### Stability Criteria

For explicit schemes (FTCS):
$$d = \frac{D_{O_2} \Delta t}{\Delta x^2} < 0.5$$

For the given default parameters:
- Critical time step: $\Delta t_{crit} = \frac{0.5 \Delta x^2}{D_{O_2}} \approx 3.64 \times 10^{-4}$ s
- Recommended: $\Delta t < 3 \times 10^{-4}$ s for FTCS

## Project Structure

```
mangala-bulan/
├── src/mangala_bulan/
│   ├── core/              # Numerical solvers
│   │   └── solver.py       # Main PDE solver implementation
│   ├── io/                # Input/output handling
│   │   ├── config_manager.py
│   │   └── data_handler.py
│   ├── utils/             # Utilities
│   │   └── logger.py      # Simulation logging
│   └── visualization/     # Plotting and animation
│       └── animator.py
├── configs/               # Scenario configurations
│   ├── ftcs_*.txt
│   ├── crank_*.txt
│   └── ...
├── outputs/              # Simulation results
└── logs/                 # Computation logs
```

## Mathematical Background

The system models oxygen transport in muscle tissue where:

1. **Fick's Diffusion**: $J = -D \nabla C$ governs molecular transport
2. **Michaelis-Menten Kinetics**: Models enzymatic oxygen consumption with saturation
3. **Myoglobin Facilitation**: Enhanced transport via reversible oxygen binding
4. **Hill Equation**: Describes cooperative oxygen binding to myoglobin

The dimensionless Damköhler number characterizes the reaction-diffusion balance:
$$Da = \frac{V_{max} L^2}{D_{O_2} K_m}$$


## Authors

- **Sandy H. S. Herho** 
- **Gandhi Napitupulu**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## Citation

If you use mangala-bulan in your research, please cite:

### Software Citation

```bibtex
@software{mangala_bulan_2024,
  title = {{\texttt{mangala-bulan}: 1D Idealized Oxygen Diffusion Solver}},
  author = {Herho, Sandy H. S. and Napitupulu, Gandhi},
  year = {2025},
  month = {12},
  version = {0.0.1},
  publisher = {PyPI},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/sandyherho/mangala-bulan},
  license = {MIT}
}
```
