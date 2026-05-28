"""
Saudi Arabia Climate Data for HVAC Simulation
Cities: Riyadh, Jeddah, Abha
Source: Weather Atlas, Climate-Data.org (1991-2020 averages)
"""

import numpy as np

# Monthly average HIGH temperatures (°C) — Jan to Dec
TEMP_HIGH = {
    "Riyadh": [20.2, 23.0, 27.5, 33.0, 39.4, 42.0, 43.6, 43.4, 40.0, 35.0, 27.8, 22.2],
    "Jeddah": [29.0, 29.4, 31.8, 34.0, 36.2, 37.8, 39.4, 38.8, 37.6, 35.6, 32.6, 30.0],
    "Abha":   [21.8, 23.0, 25.5, 27.8, 30.5, 32.3, 30.7, 30.0, 29.0, 26.5, 23.0, 21.1],
}

# Monthly average LOW temperatures (°C) — Jan to Dec
TEMP_LOW = {
    "Riyadh": [ 9.0, 10.8, 14.5, 19.5, 24.8, 26.5, 28.8, 28.4, 25.0, 20.5, 14.5, 10.6],
    "Jeddah": [20.3, 20.0, 21.4, 22.5, 23.5, 24.0, 27.0, 28.0, 26.5, 25.0, 23.0, 21.5],
    "Abha":   [12.4, 13.0, 14.5, 16.0, 19.0, 20.5, 19.5, 19.0, 17.5, 14.5, 12.0,  7.8],
}

# Monthly average MEAN temperatures (°C) — Jan to Dec
TEMP_MEAN = {
    city: [(h + l) / 2 for h, l in zip(TEMP_HIGH[city], TEMP_LOW[city])]
    for city in TEMP_HIGH
}

# Solar radiation (W/m²) — monthly average daily peak
SOLAR_RADIATION = {
    "Riyadh": [450, 520, 600, 680, 750, 800, 790, 760, 700, 600, 500, 430],
    "Jeddah": [480, 540, 620, 700, 760, 780, 770, 750, 710, 620, 520, 460],
    "Abha":   [500, 560, 630, 680, 720, 740, 680, 660, 650, 580, 510, 470],
}

# Relative humidity (%) — monthly average
HUMIDITY = {
    "Riyadh": [47, 36, 32, 27, 16, 10,  9, 11, 14, 20, 35, 47],
    "Jeddah": [56, 54, 53, 52, 48, 42, 43, 50, 57, 57, 58, 58],
    "Abha":   [57, 50, 45, 42, 32, 25, 38, 45, 35, 30, 40, 50],
}

# Climate classification
CLIMATE_CLASS = {
    "Riyadh": "BWh (Hot desert)",
    "Jeddah": "BWh (Hot coastal desert)",
    "Abha":   "BWk/Cwa (Mild highland)",
}

# Elevation (m)
ELEVATION = {"Riyadh": 612, "Jeddah": 12, "Abha": 2270}

# Annual cooling degree days (base 18°C) and heating degree days
CDD_18 = {"Riyadh": 3050, "Jeddah": 3650, "Abha": 800}
HDD_18 = {"Riyadh": 150,  "Jeddah": 0,    "Abha": 450}


def generate_hourly_temperature(city: str, month: int, day_count: int = 1) -> np.ndarray:
    """Generate synthetic hourly outdoor temperature profile.
    
    Uses sinusoidal diurnal pattern: T(h) = T_mean + amplitude * sin(2π(h-15)/24)
    where peak at 15:00 (3 PM) and trough at 03:00 (3 AM).
    
    Args:
        city: One of 'Riyadh', 'Jeddah', 'Abha'
        month: 1-12
        day_count: Number of days to generate
        
    Returns:
        np.ndarray of shape (day_count * 24,) with hourly temperatures in °C
    """
    idx = month - 1
    t_high = TEMP_HIGH[city][idx]
    t_low = TEMP_LOW[city][idx]
    t_mean = (t_high + t_low) / 2
    amplitude = (t_high - t_low) / 2
    
    hours = np.arange(day_count * 24)
    # Peak at 15:00 (hour 15), trough at 03:00 (hour 3)
    temps = t_mean + amplitude * np.sin(2 * np.pi * (hours % 24 - 15) / 24)
    
    # Add small random noise (±0.5°C) for realism
    rng = np.random.default_rng(seed=42 + month)
    temps += rng.normal(0, 0.5, size=temps.shape)
    
    return temps


def generate_annual_profile(city: str, dt_minutes: int = 15) -> dict:
    """Generate full annual temperature profile at given timestep.
    
    Args:
        city: One of 'Riyadh', 'Jeddah', 'Abha'
        dt_minutes: Timestep in minutes (default 15)
        
    Returns:
        dict with 'time_hours', 'temperature_C', 'month_index'
    """
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    steps_per_hour = 60 // dt_minutes
    
    all_temps = []
    all_months = []
    
    for m in range(12):
        n_days = days_per_month[m]
        hourly = generate_hourly_temperature(city, m + 1, n_days)
        # Interpolate to sub-hourly if needed
        if steps_per_hour > 1:
            fine = np.interp(
                np.linspace(0, len(hourly) - 1, len(hourly) * steps_per_hour),
                np.arange(len(hourly)),
                hourly
            )
        else:
            fine = hourly
        all_temps.append(fine)
        all_months.append(np.full(len(fine), m + 1))
    
    temps = np.concatenate(all_temps)
    months = np.concatenate(all_months)
    time_h = np.arange(len(temps)) * (dt_minutes / 60)
    
    return {
        "time_hours": time_h,
        "temperature_C": temps,
        "month_index": months,
        "city": city,
        "dt_minutes": dt_minutes,
    }


if __name__ == "__main__":
    for city in ["Riyadh", "Jeddah", "Abha"]:
        print(f"\n=== {city} ({CLIMATE_CLASS[city]}, {ELEVATION[city]}m) ===")
        print(f"  Annual mean: {np.mean(TEMP_MEAN[city]):.1f}°C")
        print(f"  Peak summer: {max(TEMP_HIGH[city]):.1f}°C")
        print(f"  Coldest winter: {min(TEMP_LOW[city]):.1f}°C")
        print(f"  CDD(18): {CDD_18[city]}, HDD(18): {HDD_18[city]}")
        print(f"  Monthly means: {[f'{t:.1f}' for t in TEMP_MEAN[city]]}")
