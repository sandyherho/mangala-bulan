# `mangala-bulan` 1D Idealized Oxygen Diffusion Solver

Oxygen diffusion solver with Michaelis-Menten kinetics and myoglobin facilitated diffusion.

## Authors
- Sandy H. S. Herho
- Gandhi Napitupulu

## Features
- Four numerical schemes: FTCS, DuFort-Frankel, Crank-Nicolson, Laasonen
- Michaelis-Menten oxygen consumption
- Myoglobin facilitated diffusion
- Parallel processing with Numba
- NetCDF, CSV, and animated GIF outputs

## Installation
```bash
cd mangala-bulan
pip install -e .
```

## Usage
```bash
# Run all scenarios
mangala-bulan --all

# Run specific method
mangala-bulan --method ftcs

# Run single config
mangala-bulan configs/ftcs_scenario1.txt
```
