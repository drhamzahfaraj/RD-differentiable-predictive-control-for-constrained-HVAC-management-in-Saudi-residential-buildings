"""
RD-DPC HVAC Controller: Multi-zone simulation with baselines and EnergyPlus comparison.

Simulates 3 control strategies across 3 Saudi cities for a full year:
  1. Thermostat (On/Off bang-bang) — baseline
  2. Nominal DPC — learned policy on nominal RC model
  3. RD-DPC — learned policy on residual-corrected model

Compares RC model outputs against EnergyPlus-equivalent reference
and computes relative error (epsilon).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import json
from building_model import (
    create_saudi_villa, MultiZoneRCModel, ASHRAE_COMFORT, SBC_COMFORT,
    relative_error, comfort_violation, ThermalZone
)
from saudi_climate import (
    generate_hourly_temperature, TEMP_HIGH, TEMP_LOW, TEMP_MEAN,
    SOLAR_RADIATION, CLIMATE_CLASS, ELEVATION, CDD_18, HDD_18
)


# ============================================================
# Occupancy Schedule (Saudi residential)
# ============================================================

def occupancy_schedule(hour: int) -> float:
    """Typical Saudi residential occupancy pattern.
    High occupancy evenings/nights due to hot daytime climate.
    """
    if 0 <= hour < 6:    return 0.8   # Sleeping
    elif 6 <= hour < 8:  return 1.0   # Morning routine
    elif 8 <= hour < 14: return 0.3   # Work/school
    elif 14 <= hour < 16: return 0.6  # Midday rest (common in KSA)
    elif 16 <= hour < 20: return 0.5  # Afternoon
    elif 20 <= hour < 23: return 1.0  # Evening (peak)
    else:                return 0.8   # Late night


def solar_profile(hour: int, I_peak: float) -> float:
    """Simple solar radiation profile (W/m²)."""
    if 6 <= hour <= 18:
        return I_peak * np.sin(np.pi * (hour - 6) / 12)
    return 0.0


# ============================================================
# Controller 1: Thermostat (On/Off Bang-Bang) — Baseline
# ============================================================

def thermostat_controller(T_zones: np.ndarray, setpoint: float = 24.0,
                          deadband: float = 1.0, T_out: float = 35.0) -> np.ndarray:
    """Simple on/off thermostat with deadband.
    
    Cooling: ON when T > setpoint + deadband/2
             OFF when T < setpoint - deadband/2
    Heating: ON when T < heating_setpoint - deadband/2
    """
    n = len(T_zones)
    u = np.zeros(n)
    
    heating_setpoint = 21.0
    
    for i in range(n):
        if T_zones[i] > setpoint + deadband / 2:
            u[i] = 1.0  # Full cooling
        elif T_zones[i] < heating_setpoint - deadband / 2:
            u[i] = -1.0  # Full heating
        # else: off (hysteresis)
    
    return u


# ============================================================
# Controller 2: Nominal DPC (simplified learned policy)
# ============================================================

def nominal_dpc_controller(T_zones: np.ndarray, T_out: float, hour: int,
                           setpoint: float = 24.0) -> np.ndarray:
    """Simulated nominal DPC policy.
    
    Approximates a learned neural policy trained on the nominal RC model.
    Uses proportional control with time-of-day awareness.
    """
    n = len(T_zones)
    u = np.zeros(n)
    
    heating_setpoint = 21.0
    
    for i in range(n):
        error = T_zones[i] - setpoint
        
        if error > 0:  # Needs cooling
            # Proportional with pre-cooling during off-peak
            gain = 0.3 if (0 <= hour < 6 or 22 <= hour < 24) else 0.5
            u[i] = np.clip(gain * error, 0, 1.0)
        elif T_zones[i] < heating_setpoint:
            error_heat = heating_setpoint - T_zones[i]
            u[i] = -np.clip(0.4 * error_heat, 0, 1.0)
    
    return u


# ============================================================
# Controller 3: RD-DPC (residual-corrected policy)
# ============================================================

def rddpc_controller(T_zones: np.ndarray, T_out: float, hour: int,
                     setpoint: float = 24.0, I_solar: float = 0.0) -> np.ndarray:
    """Simulated RD-DPC policy.
    
    Approximates a learned neural policy trained on the residual-corrected model.
    Key improvements over nominal DPC:
      1. Anticipatory pre-cooling before peak hours
      2. Solar-gain-aware modulation
      3. Inter-zone coordination
      4. Thermal mass exploitation
    """
    n = len(T_zones)
    u = np.zeros(n)
    
    heating_setpoint = 21.0
    T_mean = np.mean(T_zones)
    
    # Anticipatory factor: pre-cool 2h before peak
    peak_anticipation = 1.0
    if 11 <= hour <= 13:  # Pre-cooling window
        peak_anticipation = 1.3
    elif 14 <= hour <= 17:  # Peak hours — maintain
        peak_anticipation = 1.1
    elif 0 <= hour <= 5:  # Night — exploit thermal mass
        peak_anticipation = 0.6
    
    # Solar-aware gain
    solar_factor = 1.0 + 0.2 * (I_solar / 800) if I_solar > 0 else 1.0
    
    for i in range(n):
        error = T_zones[i] - setpoint
        
        if error > -0.5:  # Start cooling slightly before setpoint
            # Proportional-predictive control
            effective_error = max(0, error + 0.5)  # Offset for pre-cooling
            gain = 0.35 * peak_anticipation * solar_factor
            u[i] = np.clip(gain * effective_error, 0, 0.95)
        
        if T_zones[i] < heating_setpoint:
            error_heat = heating_setpoint - T_zones[i]
            u[i] = -np.clip(0.35 * error_heat, 0, 0.9)
        
        # Inter-zone coordination: reduce if neighbors are cool
        if i > 0 and T_zones[i-1] < setpoint - 1:
            u[i] *= 0.85
    
    return u


# ============================================================
# EnergyPlus Reference Generator
# ============================================================

def energyplus_reference(T_zones_rc: np.ndarray, T_out: float,
                         noise_std: float = 0.3) -> np.ndarray:
    """Generate EnergyPlus-equivalent reference temperatures.
    
    Applies systematic corrections to RC model outputs to simulate
    the higher-fidelity EnergyPlus model:
      - Wall thermal mass lag (0.5-2h delay)
      - Radiant exchange effects
      - Infiltration
      - Humidity effects (latent load)
    
    In practice, this would come from actual EnergyPlus co-simulation.
    Here we simulate the expected discrepancy for demonstration.
    
    Args:
        T_zones_rc: RC model temperatures
        T_out: Outdoor temperature
        noise_std: Standard deviation of model discrepancy (°C)
    
    Returns:
        T_zones_ep: EnergyPlus-equivalent temperatures
    """
    # Systematic bias: RC model slightly underestimates thermal lag
    bias = 0.15 * (T_out - 30) / 15  # More bias at extreme temperatures
    
    # Random component representing unmodeled effects
    rng = np.random.default_rng(seed=int(abs(T_out * 100)))
    noise = rng.normal(0, noise_std, size=T_zones_rc.shape)
    
    # Infiltration effect: slight warming in cooling season
    infiltration = 0.1 * max(0, T_out - 24) / 20
    
    T_ep = T_zones_rc + bias + noise + infiltration
    
    return T_ep


# ============================================================
# Full Annual Simulation
# ============================================================

def simulate_annual(city: str, thermal_mass: str = "medium",
                    dt_minutes: int = 15) -> dict:
    """Run full annual simulation for one city.
    
    Args:
        city: 'Riyadh', 'Jeddah', or 'Abha'
        thermal_mass: 'low', 'medium', or 'high'
        dt_minutes: Timestep (minutes)
        
    Returns:
        dict with comprehensive results
    """
    # Create building
    villa = create_saudi_villa()
    for zone in villa:
        zone.thermal_mass = thermal_mass
    
    model = MultiZoneRCModel(villa, dt_seconds=dt_minutes * 60)
    n_zones = len(villa)
    
    # Days per month
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    steps_per_hour = 60 // dt_minutes
    
    # Controllers
    controllers = {
        "Thermostat": thermostat_controller,
        "Nominal_DPC": nominal_dpc_controller,
        "RD_DPC": rddpc_controller,
    }
    
    results = {ctrl: {
        "T_zones": [],
        "T_ep": [],
        "u_hvac": [],
        "power_kw": [],
        "energy_kwh": [],
        "comfort_violations": 0,
        "total_steps": 0,
    } for ctrl in controllers}
    
    for month_idx in range(12):
        month = month_idx + 1
        n_days = days_per_month[month_idx]
        I_solar_peak = SOLAR_RADIATION[city][month_idx]
        
        # Generate hourly outdoor temperature
        T_out_hourly = generate_hourly_temperature(city, month, n_days)
        
        for ctrl_name in controllers:
            # Initial temperature: 28°C (typical Saudi indoor without AC)
            T_zones = np.full(n_zones, 28.0)
            
            for day in range(n_days):
                for step in range(24 * steps_per_hour):
                    hour_idx = day * 24 + step // steps_per_hour
                    hour_of_day = (step // steps_per_hour) % 24
                    
                    if hour_idx >= len(T_out_hourly):
                        hour_idx = len(T_out_hourly) - 1
                    
                    T_out = T_out_hourly[hour_idx]
                    I_solar = solar_profile(hour_of_day, I_solar_peak)
                    occ = occupancy_schedule(hour_of_day)
                    
                    # Get control action
                    if ctrl_name == "Thermostat":
                        u = thermostat_controller(T_zones, T_out=T_out)
                    elif ctrl_name == "Nominal_DPC":
                        u = nominal_dpc_controller(T_zones, T_out, hour_of_day)
                    else:
                        u = rddpc_controller(T_zones, T_out, hour_of_day, I_solar=I_solar)
                    
                    # Step building model
                    T_zones = model.step(T_zones, T_out, I_solar, u, occ)
                    
                    # EnergyPlus reference
                    T_ep = energyplus_reference(T_zones, T_out)
                    
                    # Record (sample every hour to reduce memory)
                    if step % steps_per_hour == 0:
                        results[ctrl_name]["T_zones"].append(T_zones.copy())
                        results[ctrl_name]["T_ep"].append(T_ep.copy())
                        results[ctrl_name]["u_hvac"].append(u.copy())
                        power = model.compute_power_kw(u)
                        energy = model.compute_energy_kwh(u)
                        results[ctrl_name]["power_kw"].append(power)
                        results[ctrl_name]["energy_kwh"].append(energy)
                        
                        # Comfort check
                        mode = "heating" if T_out < 18 else "cooling"
                        cv = comfort_violation(T_zones, mode)
                        results[ctrl_name]["comfort_violations"] += cv["total_violations"]
                        results[ctrl_name]["total_steps"] += 1
        
        print(f"  {city} — Month {month:2d}/12 complete")
    
    # Convert to arrays
    for ctrl_name in controllers:
        for key in ["T_zones", "T_ep", "u_hvac"]:
            results[ctrl_name][key] = np.array(results[ctrl_name][key])
        for key in ["power_kw", "energy_kwh"]:
            results[ctrl_name][key] = np.array(results[ctrl_name][key])
    
    # Compute summary statistics
    summary = {}
    for ctrl_name in controllers:
        r = results[ctrl_name]
        T_all = r["T_zones"]
        T_ep = r["T_ep"]
        
        # Relative error vs EnergyPlus
        eps = relative_error(T_all, T_ep)
        
        # Energy
        total_energy_kwh = np.sum(r["energy_kwh"])
        annual_energy_kwh_m2 = total_energy_kwh / 147  # per m²
        
        # Comfort
        comfort_rate = 1.0 - r["comfort_violations"] / (r["total_steps"] * n_zones)
        
        # Temperature stats
        T_mean_indoor = np.mean(T_all)
        T_max_indoor = np.max(T_all)
        T_min_indoor = np.min(T_all)
        
        summary[ctrl_name] = {
            "annual_energy_kwh": round(total_energy_kwh, 1),
            "energy_kwh_per_m2": round(annual_energy_kwh_m2, 1),
            "comfort_rate_pct": round(comfort_rate * 100, 2),
            "T_mean_indoor_C": round(T_mean_indoor, 2),
            "T_max_indoor_C": round(T_max_indoor, 2),
            "T_min_indoor_C": round(T_min_indoor, 2),
            "epsilon_vs_energyplus": round(eps["epsilon"], 6),
            "mae_vs_energyplus_C": round(eps["mae_C"], 3),
            "rmse_vs_energyplus_C": round(eps["rmse_C"], 3),
            "max_error_vs_energyplus_C": round(eps["max_error_C"], 3),
            "mean_power_kw": round(np.mean(r["power_kw"]), 3),
            "peak_power_kw": round(np.max(r["power_kw"]), 3),
        }
    
    # Energy savings
    E_therm = summary["Thermostat"]["annual_energy_kwh"]
    E_nom = summary["Nominal_DPC"]["annual_energy_kwh"]
    E_rd = summary["RD_DPC"]["annual_energy_kwh"]
    
    summary["savings_vs_thermostat"] = {
        "Nominal_DPC_pct": round((1 - E_nom / E_therm) * 100, 2) if E_therm > 0 else 0,
        "RD_DPC_pct": round((1 - E_rd / E_therm) * 100, 2) if E_therm > 0 else 0,
    }
    summary["savings_vs_nominal_dpc"] = {
        "RD_DPC_pct": round((1 - E_rd / E_nom) * 100, 2) if E_nom > 0 else 0,
    }
    
    return {
        "city": city,
        "climate": CLIMATE_CLASS[city],
        "elevation_m": ELEVATION[city],
        "thermal_mass": thermal_mass,
        "summary": summary,
    }


# ============================================================
# Sensitivity Analysis
# ============================================================

def sensitivity_analysis(city: str = "Riyadh") -> dict:
    """Run simulation with low/medium/high thermal mass."""
    results = {}
    for mass in ["low", "medium", "high"]:
        print(f"\n--- Sensitivity: {city}, thermal_mass={mass} ---")
        r = simulate_annual(city, thermal_mass=mass)
        results[mass] = r["summary"]
    return results


# ============================================================
# Main: Run all cities
# ============================================================

if __name__ == "__main__":
    all_results = {}
    
    for city in ["Riyadh", "Jeddah", "Abha"]:
        print(f"\n{'='*60}")
        print(f"Simulating: {city} ({CLIMATE_CLASS[city]})")
        print(f"{'='*60}")
        
        result = simulate_annual(city, thermal_mass="medium")
        all_results[city] = result
        
        print(f"\n--- {city} Results ---")
        for ctrl, stats in result["summary"].items():
            if isinstance(stats, dict) and "annual_energy_kwh" in stats:
                print(f"\n  {ctrl}:")
                print(f"    Energy:     {stats['annual_energy_kwh']:,.0f} kWh/yr")
                print(f"    Energy/m²:  {stats['energy_kwh_per_m2']:.1f} kWh/m²/yr")
                print(f"    Comfort:    {stats['comfort_rate_pct']:.1f}%")
                print(f"    T_indoor:   {stats['T_mean_indoor_C']:.1f}°C (mean)")
                print(f"    ε (EP):     {stats['epsilon_vs_energyplus']:.4f}")
                print(f"    MAE (EP):   {stats['mae_vs_energyplus_C']:.3f}°C")
        
        savings = result["summary"].get("savings_vs_thermostat", {})
        print(f"\n  Energy savings vs Thermostat:")
        print(f"    Nominal DPC: {savings.get('Nominal_DPC_pct', 0):.1f}%")
        print(f"    RD-DPC:      {savings.get('RD_DPC_pct', 0):.1f}%")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'hvac_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
