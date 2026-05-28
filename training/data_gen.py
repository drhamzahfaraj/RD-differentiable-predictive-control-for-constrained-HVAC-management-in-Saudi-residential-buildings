"""
Data Generator for RD-DPC Training
Generates training scenarios from Saudi climate profiles.
"""

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'data'))
from saudi_climate import generate_hourly_temperature, SOLAR_RADIATION, TEMP_HIGH, TEMP_LOW


def occupancy_schedule(hour):
    """Saudi residential occupancy pattern."""
    schedule = [0.8,0.8,0.8,0.8,0.8,0.8, 1.0,1.0, 0.3,0.3,0.3,0.3,0.3,0.3,
                0.6,0.6, 0.5,0.5,0.5,0.5, 1.0,1.0,1.0, 0.8]
    return schedule[int(hour) % 24]


def solar_profile(hour, I_peak):
    """Solar radiation W/m² at given hour."""
    h = hour % 24
    if 6 <= h <= 18:
        return I_peak * np.sin(np.pi * (h - 6) / 12)
    return 0.0


def generate_scenarios(city, n_scenarios=1000, horizon=24, month=None, seed=42):
    """Generate training scenarios for DPC.
    
    Each scenario is a sequence of (T_out, I_sol, hour, month) over the prediction horizon.
    
    Args:
        city: 'Riyadh', 'Jeddah', or 'Abha'
        n_scenarios: Number of scenarios to generate
        horizon: Prediction horizon (number of 15-min steps) — default 24 = 6 hours
        month: Specific month (1-12) or None for random
        seed: Random seed
        
    Returns:
        scenarios: dict with torch tensors
            'T_out': (n_scenarios, horizon) outdoor temps
            'I_sol': (n_scenarios, horizon) solar irradiance
            'hour': (n_scenarios, horizon) hour of day
            'month': (n_scenarios, horizon) month index
            'occupancy': (n_scenarios, horizon) occupancy
            'T_init': (n_scenarios, 5) initial zone temperatures
            'E_cum_init': (n_scenarios, 1) initial cumulative energy
    """
    rng = np.random.default_rng(seed)
    
    T_out_all = np.zeros((n_scenarios, horizon))
    I_sol_all = np.zeros((n_scenarios, horizon))
    hour_all = np.zeros((n_scenarios, horizon))
    month_all = np.zeros((n_scenarios, horizon))
    occ_all = np.zeros((n_scenarios, horizon))
    
    for s in range(n_scenarios):
        # Random month and starting hour
        if month is None:
            m = rng.integers(1, 13)
        else:
            m = month
        start_hour = rng.uniform(0, 24)
        
        # Generate outdoor temperature for this scenario
        T_hourly = generate_hourly_temperature(city, m, day_count=2)
        I_peak = SOLAR_RADIATION[city][m - 1]
        
        for k in range(horizon):
            h = (start_hour + k * 0.25) % 24  # 15-min steps
            hour_idx = int(h) % len(T_hourly)
            
            # Interpolate temperature
            h_frac = h - int(h)
            T1 = T_hourly[hour_idx % len(T_hourly)]
            T2 = T_hourly[(hour_idx + 1) % len(T_hourly)]
            T_out_all[s, k] = T1 * (1 - h_frac) + T2 * h_frac
            
            I_sol_all[s, k] = solar_profile(h, I_peak)
            hour_all[s, k] = h
            month_all[s, k] = m
            occ_all[s, k] = occupancy_schedule(h)
    
    # Initial zone temperatures: random within comfort zone ± 2°C
    T_init = rng.uniform(21.0, 26.0, size=(n_scenarios, 5))
    
    # Initial cumulative energy: random 0-5000 kWh (within billing month)
    E_cum_init = rng.uniform(0, 5000, size=(n_scenarios, 1))
    
    return {
        'T_out': torch.tensor(T_out_all, dtype=torch.float32),
        'I_sol': torch.tensor(I_sol_all, dtype=torch.float32),
        'hour': torch.tensor(hour_all, dtype=torch.float32),
        'month': torch.tensor(month_all, dtype=torch.float32),
        'occupancy': torch.tensor(occ_all, dtype=torch.float32),
        'T_init': torch.tensor(T_init, dtype=torch.float32),
        'E_cum_init': torch.tensor(E_cum_init, dtype=torch.float32),
    }


def build_initial_state(T_init, T_out_0, I_sol_0, E_cum, hour_0, month_0):
    """Construct the 12-dim state vector from components.
    
    Args:
        T_init: (batch, 5) zone temperatures
        T_out_0: (batch,) outdoor temperature
        I_sol_0: (batch,) solar irradiance
        E_cum: (batch, 1) cumulative energy
        hour_0: (batch,) hour of day
        month_0: (batch,) month
        
    Returns:
        x0: (batch, 12) state vector
    """
    sin_h = torch.sin(2 * np.pi * hour_0 / 24).unsqueeze(1)
    cos_h = torch.cos(2 * np.pi * hour_0 / 24).unsqueeze(1)
    sin_m = torch.sin(2 * np.pi * month_0 / 12).unsqueeze(1)
    cos_m = torch.cos(2 * np.pi * month_0 / 12).unsqueeze(1)
    
    x0 = torch.cat([
        T_init,                        # (batch, 5)
        T_out_0.unsqueeze(1),          # (batch, 1)
        I_sol_0.unsqueeze(1),          # (batch, 1)
        E_cum,                         # (batch, 1)
        sin_h, cos_h, sin_m, cos_m,   # (batch, 4)
    ], dim=1)
    
    return x0


def update_exogenous(x, T_out_k, I_sol_k, hour_k, month_k):
    """Update exogenous variables in state (T_out, I_sol, time features).
    
    Args:
        x: (batch, 12) current state
        T_out_k, I_sol_k, hour_k, month_k: (batch,) new values
        
    Returns:
        x_updated: (batch, 12) with updated exogenous variables
    """
    x_new = x.clone()
    x_new[:, 5] = T_out_k
    x_new[:, 6] = I_sol_k
    x_new[:, 8] = torch.sin(2 * np.pi * hour_k / 24)
    x_new[:, 9] = torch.cos(2 * np.pi * hour_k / 24)
    x_new[:, 10] = torch.sin(2 * np.pi * month_k / 12)
    x_new[:, 11] = torch.cos(2 * np.pi * month_k / 12)
    return x_new
