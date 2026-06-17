# -*- coding: utf-8 -*-
"""
Hamiltonian Dataset Generator for Physics-Informed Neural Networks.
Part of the "On-Demand Adaptive PID Control via Neural Networks" framework.

This script generates a synthetic dataset of robot velocities and their 
corresponding theoretical canonical Hamiltonian (kinetic energy) based on the 
physical properties of a 4-wheeled omnidirectional robot. This dataset 
is used to pre-train the Hamiltonian prediction model.
"""

import numpy as np
import pandas as pd

# --- Robot Physical Parameters ---
# Geometry [m]
L = 0.2355
l = 0.15
R_WHEEL = 0.0475

# Mass [kg]
M_BASE = 20.0
M_WHEEL = 1.0
M_TOTAL_CONST = M_BASE + (4.0 * M_WHEEL)  # Total mass without payload

# Moment of Inertia [kg*m^2]
IZZ_CHASSIS = (1.0 / 3.0) * M_BASE * (L**2 + l**2)
IZZ_WHEEL   = M_WHEEL * (L**2 + l**2) + 0.25 * M_WHEEL * (R_WHEEL**2)
IZZ_CONST   = IZZ_CHASSIS + (4.0 * IZZ_WHEEL)

def calculate_true_hamiltonian(ux, uy, uth, m_tot=M_TOTAL_CONST, izz=IZZ_CONST):
    """Calculates the theoretical kinetic energy (Hamiltonian) of the system."""
    return 0.5 * m_tot * (ux**2 + uy**2) + 0.5 * izz * (uth**2)

def generate_dataset(
    n_samples=500000,
    ux_range=(-1.5, 1.5),
    uy_range=(-1.5, 1.5),
    uth_range=(-1.5, 1.5),
    seed=42,
    out_csv="hamiltonian_dataset.csv"
):
    """
    Generates uniformly distributed velocities and computes their Hamiltonian.
    Saves the resulting dataset to a CSV file for neural network training.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random velocity samples
    ux  = rng.uniform(*ux_range, size=n_samples)
    uy  = rng.uniform(*uy_range, size=n_samples)
    uth = rng.uniform(*uth_range, size=n_samples)
    
    # Compute canonical Hamiltonian
    H = calculate_true_hamiltonian(ux, uy, uth)
    
    # Create DataFrame and save
    df = pd.DataFrame({
        "ux": ux, 
        "uy": uy, 
        "uth": uth, 
        "H": H
    })
    
    df.to_csv(out_csv, index=False)
    print(f"[OK] Dataset successfully saved to: {out_csv} | Samples: {len(df)}")

if __name__ == "__main__":
    print("[INFO] Initializing Hamiltonian dataset generation...")
    generate_dataset()