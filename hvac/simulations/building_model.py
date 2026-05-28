"""
Multi-Zone Residential Building Model for Saudi Arabia
RC (Resistance-Capacitance) thermal network with 3 HVAC unit sizes.

Building: 5-zone Saudi residential villa
  Zone 1 (Bedroom 1):    20 m² — Small AC (12,600 BTU / 3.69 kW)
  Zone 2 (Bedroom 2):    22 m² — Small AC (12,600 BTU / 3.69 kW)
  Zone 3 (Living Room):  40 m² — Large AC (21,000 BTU / 6.15 kW)
  Zone 4 (Kitchen/Dining):30 m² — Medium AC (18,500 BTU / 5.42 kW)
  Zone 5 (Master Bed):   35 m² — Medium AC (18,500 BTU / 5.42 kW)

Total floor area: 147 m² (typical Saudi residential villa)
Ceiling height: 3.0 m (Saudi building code standard)

HVAC Units (T3-rated split AC, On/Off compressor):
  Small:  12,600 BTU/h = 3.69 kW cooling, EER ≈ 3.2, rated power ≈ 1.15 kW
  Medium: 18,500 BTU/h = 5.42 kW cooling, EER ≈ 3.0, rated power ≈ 1.81 kW
  Large:  21,000 BTU/h = 6.15 kW cooling, EER ≈ 2.9, rated power ≈ 2.12 kW

All units: Heat & Cool, On/Off control, T3 Rotary Compressor, R-410A refrigerant
T3 rating: certified for outdoor temperatures up to 52°C (critical for Saudi summers)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# HVAC Unit Specifications
# ============================================================

@dataclass
class HVACUnit:
    """Split AC unit specifications (without brand names)."""
    name: str
    capacity_btu: float        # BTU/h
    capacity_kw: float         # kW thermal
    eer: float                 # Energy Efficiency Ratio (BTU/Wh)
    rated_power_kw: float      # Electrical power draw (kW)
    cop_heating: float         # COP in heating mode
    heating_capacity_kw: float # Heating capacity (kW)
    t3_rated: bool = True      # T3 = operates up to 52°C outdoor
    refrigerant: str = "R-410A"
    compressor: str = "Rotary, On/Off"
    
    @property
    def capacity_w(self) -> float:
        return self.capacity_kw * 1000
    
    @property
    def rated_power_w(self) -> float:
        return self.rated_power_kw * 1000


# Three AC unit sizes matching Saudi residential market
AC_SMALL = HVACUnit(
    name="Split AC 12,600 BTU",
    capacity_btu=12600,
    capacity_kw=3.69,     # 12600 BTU/h ÷ 3412.14 = 3.69 kW
    eer=3.2,
    rated_power_kw=1.15,  # 3690 W / 3.2 EER
    cop_heating=2.8,
    heating_capacity_kw=3.2,
)

AC_MEDIUM = HVACUnit(
    name="Split AC 18,500 BTU",
    capacity_btu=18500,
    capacity_kw=5.42,     # 18500 BTU/h ÷ 3412.14 = 5.42 kW
    eer=3.0,
    rated_power_kw=1.81,  # 5420 W / 3.0 EER
    cop_heating=2.6,
    heating_capacity_kw=4.8,
)

AC_LARGE = HVACUnit(
    name="Split AC 21,000 BTU",
    capacity_btu=21000,
    capacity_kw=6.15,     # 21000 BTU/h ÷ 3412.14 = 6.15 kW
    eer=2.9,
    rated_power_kw=2.12,  # 6150 W / 2.9 EER
    cop_heating=2.5,
    heating_capacity_kw=5.5,
)


# ============================================================
# Thermal Zone Model
# ============================================================

@dataclass
class ThermalZone:
    """Single thermal zone with RC model parameters."""
    name: str
    area_m2: float                    # Floor area (m²)
    height_m: float = 3.0            # Ceiling height (m)
    hvac: HVACUnit = None            # Assigned HVAC unit
    
    # Thermal properties
    wall_R: float = 2.5              # Wall thermal resistance (m²·K/W) — insulated concrete block
    roof_R: float = 3.0              # Roof thermal resistance (m²·K/W)
    window_U: float = 3.5            # Window U-value (W/m²·K) — double glazed
    window_ratio: float = 0.15       # Window-to-wall ratio
    
    # Thermal mass scenarios
    thermal_mass: str = "medium"     # "low", "medium", "high"
    
    # Internal gains
    occupancy_W: float = 75.0        # Per person (W)
    n_occupants: float = 1.5         # Average occupants
    equipment_W_per_m2: float = 5.0  # Equipment (W/m²)
    lighting_W_per_m2: float = 8.0   # Lighting (W/m²)
    
    # Adjacent zone coupling
    adjacent_zones: List[int] = field(default_factory=list)
    inter_zone_UA: float = 10.0      # Inter-zone conductance (W/K)
    
    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.height_m
    
    @property
    def thermal_capacitance_J_K(self) -> float:
        """Effective thermal capacitance based on mass scenario."""
        # Air only: ρ_air × c_air × V = 1.2 × 1005 × V
        C_air = 1.2 * 1005 * self.volume_m3
        
        multipliers = {
            "low": 1.0,      # Air only (unrealistically fast — used for sensitivity)
            "medium": 15.0,  # Air + furniture + light walls (~typical)
            "high": 40.0,    # Air + concrete walls + heavy furnishings
        }
        return C_air * multipliers[self.thermal_mass]
    
    @property
    def envelope_UA_W_K(self) -> float:
        """Overall envelope UA value (W/K)."""
        perimeter = 4 * np.sqrt(self.area_m2)  # Approximate square room
        wall_area = perimeter * self.height_m
        window_area = wall_area * self.window_ratio
        opaque_area = wall_area - window_area
        roof_area = self.area_m2
        
        UA_wall = opaque_area / self.wall_R
        UA_window = window_area * self.window_U
        UA_roof = roof_area / self.roof_R
        
        return UA_wall + UA_window + UA_roof
    
    @property
    def internal_gains_W(self) -> float:
        """Total internal heat gains (W)."""
        Q_occ = self.occupancy_W * self.n_occupants
        Q_equip = self.equipment_W_per_m2 * self.area_m2
        Q_light = self.lighting_W_per_m2 * self.area_m2
        return Q_occ + Q_equip + Q_light
    
    @property
    def solar_gain_coefficient(self) -> float:
        """Solar heat gain through windows (fraction)."""
        perimeter = 4 * np.sqrt(self.area_m2)
        window_area = perimeter * self.height_m * self.window_ratio
        SHGC = 0.4  # Solar Heat Gain Coefficient for double glazing
        return window_area * SHGC


# ============================================================
# Multi-Zone Building
# ============================================================

def create_saudi_villa() -> List[ThermalZone]:
    """Create 5-zone Saudi residential villa."""
    zones = [
        ThermalZone(
            name="Bedroom 1", area_m2=20.0, hvac=AC_SMALL,
            n_occupants=1.0, equipment_W_per_m2=3.0, lighting_W_per_m2=6.0,
            adjacent_zones=[1, 2],
        ),
        ThermalZone(
            name="Bedroom 2", area_m2=22.0, hvac=AC_SMALL,
            n_occupants=1.0, equipment_W_per_m2=3.0, lighting_W_per_m2=6.0,
            adjacent_zones=[0, 3],
        ),
        ThermalZone(
            name="Living Room", area_m2=40.0, hvac=AC_LARGE,
            n_occupants=3.0, equipment_W_per_m2=8.0, lighting_W_per_m2=10.0,
            window_ratio=0.25,  # More windows in living room
            adjacent_zones=[0, 3, 4],
        ),
        ThermalZone(
            name="Kitchen/Dining", area_m2=30.0, hvac=AC_MEDIUM,
            n_occupants=2.0, equipment_W_per_m2=15.0, lighting_W_per_m2=12.0,
            adjacent_zones=[1, 2, 4],
        ),
        ThermalZone(
            name="Master Bedroom", area_m2=35.0, hvac=AC_MEDIUM,
            n_occupants=2.0, equipment_W_per_m2=5.0, lighting_W_per_m2=8.0,
            adjacent_zones=[2, 3],
        ),
    ]
    return zones


# ============================================================
# RC Thermal Simulation
# ============================================================

class MultiZoneRCModel:
    """Multi-zone RC thermal model with inter-zone coupling.
    
    State equation for each zone i:
        C_i * dT_i/dt = UA_i * (T_out - T_i) 
                      + Σ_j UA_ij * (T_j - T_i)      [inter-zone]
                      + Q_internal_i                    [internal gains]
                      + A_solar_i * I_solar             [solar gains]
                      - Q_hvac_i * u_i                  [HVAC: +cooling/-heating]
    
    Discretized with Euler method at timestep dt.
    """
    
    def __init__(self, zones: List[ThermalZone], dt_seconds: float = 900.0):
        self.zones = zones
        self.n_zones = len(zones)
        self.dt = dt_seconds  # 15-minute timestep
        
    def step(self, T_zones: np.ndarray, T_out: float, I_solar: float,
             u_hvac: np.ndarray, occupancy_schedule: float = 1.0) -> np.ndarray:
        """Advance one timestep.
        
        Args:
            T_zones: Current zone temperatures (°C), shape (n_zones,)
            T_out: Outdoor temperature (°C)
            I_solar: Solar radiation (W/m²)
            u_hvac: HVAC control signals, shape (n_zones,)
                    u > 0: cooling mode (fraction of capacity)
                    u < 0: heating mode (fraction of capacity)
                    u = 0: off
            occupancy_schedule: Multiplier for internal gains (0-1)
            
        Returns:
            T_zones_next: Updated zone temperatures (°C), shape (n_zones,)
        """
        T_next = T_zones.copy()
        
        for i, zone in enumerate(self.zones):
            C = zone.thermal_capacitance_J_K
            
            # Envelope heat transfer
            Q_env = zone.envelope_UA_W_K * (T_out - T_zones[i])
            
            # Inter-zone coupling
            Q_adj = 0.0
            for j in zone.adjacent_zones:
                Q_adj += zone.inter_zone_UA * (T_zones[j] - T_zones[i])
            
            # Internal gains (scaled by occupancy)
            Q_int = zone.internal_gains_W * occupancy_schedule
            
            # Solar gains
            Q_solar = zone.solar_gain_coefficient * I_solar
            
            # HVAC
            Q_hvac = 0.0
            if u_hvac[i] > 0:  # Cooling
                Q_hvac = -zone.hvac.capacity_w * u_hvac[i]
            elif u_hvac[i] < 0:  # Heating
                Q_hvac = zone.hvac.heating_capacity_kw * 1000 * abs(u_hvac[i])
            
            # Energy balance
            dT = (Q_env + Q_adj + Q_int + Q_solar + Q_hvac) / C * self.dt
            T_next[i] = T_zones[i] + dT
        
        return T_next
    
    def compute_power_kw(self, u_hvac: np.ndarray) -> float:
        """Compute total electrical power consumption (kW)."""
        total = 0.0
        for i, zone in enumerate(self.zones):
            if u_hvac[i] > 0:  # Cooling
                total += zone.hvac.rated_power_kw * u_hvac[i]
            elif u_hvac[i] < 0:  # Heating
                heat_power = zone.hvac.heating_capacity_kw * abs(u_hvac[i]) / zone.hvac.cop_heating
                total += heat_power
        return total
    
    def compute_energy_kwh(self, u_hvac: np.ndarray) -> float:
        """Energy for one timestep (kWh)."""
        return self.compute_power_kw(u_hvac) * (self.dt / 3600)


# ============================================================
# ASHRAE 55 Thermal Comfort
# ============================================================

ASHRAE_COMFORT = {
    "cooling_min": 23.0,   # °C — lower bound summer comfort
    "cooling_max": 26.0,   # °C — upper bound summer comfort
    "heating_min": 20.0,   # °C — lower bound winter comfort
    "heating_max": 23.5,   # °C — upper bound winter comfort
    "acceptable_min": 20.0,  # °C — absolute minimum
    "acceptable_max": 26.0,  # °C — absolute maximum
}

# Saudi-specific: SBC 601 allows wider range for energy conservation
SBC_COMFORT = {
    "cooling_setpoint": 24.0,  # °C — Saudi Building Code recommended
    "heating_setpoint": 21.0,  # °C
    "deadband": 1.0,           # °C — ±1°C around setpoint
}


def comfort_violation(T_zones: np.ndarray, mode: str = "cooling") -> dict:
    """Check ASHRAE 55 comfort compliance.
    
    Args:
        T_zones: Zone temperatures (°C)
        mode: 'cooling' or 'heating'
        
    Returns:
        dict with violation counts and statistics
    """
    if mode == "cooling":
        too_hot = np.sum(T_zones > ASHRAE_COMFORT["cooling_max"])
        too_cold = np.sum(T_zones < ASHRAE_COMFORT["cooling_min"])
        target = (ASHRAE_COMFORT["cooling_min"] + ASHRAE_COMFORT["cooling_max"]) / 2
    else:
        too_hot = np.sum(T_zones > ASHRAE_COMFORT["heating_max"])
        too_cold = np.sum(T_zones < ASHRAE_COMFORT["heating_min"])
        target = (ASHRAE_COMFORT["heating_min"] + ASHRAE_COMFORT["heating_max"]) / 2
    
    return {
        "too_hot": int(too_hot),
        "too_cold": int(too_cold),
        "total_violations": int(too_hot + too_cold),
        "mean_deviation": float(np.mean(np.abs(T_zones - target))),
        "max_deviation": float(np.max(np.abs(T_zones - target))),
    }


# ============================================================
# Relative Error Calculation
# ============================================================

def relative_error(T_rc: np.ndarray, T_reference: np.ndarray) -> dict:
    """Compute relative error between RC model and reference (e.g., EnergyPlus).
    
    ε_rel = |T_RC - T_ref| / |T_ref|  (element-wise, averaged)
    
    Args:
        T_rc: RC model temperatures (°C), shape (n_steps, n_zones)
        T_reference: Reference temperatures (°C), same shape
        
    Returns:
        dict with error metrics
    """
    abs_error = np.abs(T_rc - T_reference)
    # Use Kelvin for relative error to avoid division by near-zero
    T_ref_K = T_reference + 273.15
    rel_error = abs_error / T_ref_K
    
    return {
        "mae_C": float(np.mean(abs_error)),
        "rmse_C": float(np.sqrt(np.mean(abs_error**2))),
        "max_error_C": float(np.max(abs_error)),
        "mean_relative_error": float(np.mean(rel_error)),
        "max_relative_error": float(np.max(rel_error)),
        "epsilon": float(np.mean(rel_error)),  # ε notation
    }


# ============================================================
# Running Example with Numeric Values
# ============================================================

def running_example():
    """Numeric example: 18,000 BTU split AC in 35 m² master bedroom, Riyadh July.
    
    Unit specifications (generic, no brand):
      - Cooling capacity: 18,000 BTU/h (5.27 kW)
      - Heating capacity: 19,000 BTU/h (5.57 kW) 
      - EER: 3.0
      - Rated power (cooling): 1.76 kW
      - Refrigerant: R-410A
      - Compressor: Rotary, On/Off
      - Wi-Fi enabled
      - T3 rated (operates up to 52°C outdoor)
    """
    # Unit specs (from product page, 18,000 BTU hot/cold rotary WiFi)
    ac = HVACUnit(
        name="Split AC 18,000 BTU (WiFi)",
        capacity_btu=18000,
        capacity_kw=5.27,
        eer=3.0,
        rated_power_kw=1.76,
        cop_heating=2.7,
        heating_capacity_kw=5.57,
    )
    
    # Room: Master bedroom, 35 m², Riyadh, July
    zone = ThermalZone(
        name="Master Bedroom (Example)", area_m2=35.0, hvac=ac,
        n_occupants=2.0, thermal_mass="medium",
    )
    
    print("=" * 60)
    print("RUNNING EXAMPLE: 18,000 BTU Split AC in 35 m² Room")
    print("Location: Riyadh, Saudi Arabia — July (peak summer)")
    print("=" * 60)
    print(f"\nRoom properties:")
    print(f"  Floor area:     {zone.area_m2} m²")
    print(f"  Volume:         {zone.volume_m3:.0f} m³")
    print(f"  Envelope UA:    {zone.envelope_UA_W_K:.1f} W/K")
    print(f"  Thermal cap:    {zone.thermal_capacitance_J_K/1e6:.2f} MJ/K")
    print(f"  Internal gains: {zone.internal_gains_W:.0f} W")
    print(f"  Solar coeff:    {zone.solar_gain_coefficient:.2f} m²")
    print(f"\nHVAC unit:")
    print(f"  Cooling:  {ac.capacity_btu:,} BTU/h ({ac.capacity_kw:.2f} kW)")
    print(f"  Heating:  {ac.heating_capacity_kw:.2f} kW")
    print(f"  EER:      {ac.eer}")
    print(f"  Power:    {ac.rated_power_kw:.2f} kW")
    print(f"  T3 rated: {ac.t3_rated} (up to 52°C outdoor)")
    
    # Simulate 24h in Riyadh July
    from saudi_climate import generate_hourly_temperature
    T_out_hourly = generate_hourly_temperature("Riyadh", 7, day_count=1)
    
    print(f"\nOutdoor temperature (Riyadh, July):")
    print(f"  Peak:  {max(T_out_hourly):.1f}°C at ~15:00")
    print(f"  Low:   {min(T_out_hourly):.1f}°C at ~03:00")
    print(f"  Mean:  {np.mean(T_out_hourly):.1f}°C")
    
    # Thermal balance at peak (15:00, T_out = 43.6°C, T_in = 24°C setpoint)
    T_out_peak = max(T_out_hourly)
    T_setpoint = 24.0  # SBC 601 recommended
    I_solar_peak = 790  # W/m² (July Riyadh)
    
    Q_env = zone.envelope_UA_W_K * (T_out_peak - T_setpoint)
    Q_int = zone.internal_gains_W
    Q_solar = zone.solar_gain_coefficient * I_solar_peak
    Q_total = Q_env + Q_int + Q_solar
    
    print(f"\nPeak thermal load at 15:00 (T_out={T_out_peak:.1f}°C, T_set={T_setpoint}°C):")
    print(f"  Envelope:  {Q_env:.0f} W")
    print(f"  Internal:  {Q_int:.0f} W")
    print(f"  Solar:     {Q_solar:.0f} W")
    print(f"  TOTAL:     {Q_total:.0f} W ({Q_total/1000:.2f} kW)")
    print(f"  AC capacity: {ac.capacity_kw:.2f} kW")
    print(f"  Utilization: {Q_total/ac.capacity_w*100:.1f}%")
    
    if Q_total < ac.capacity_w:
        print(f"  ✓ Unit can handle peak load (margin: {(1-Q_total/ac.capacity_w)*100:.1f}%)")
    else:
        print(f"  ✗ Unit undersized by {(Q_total/ac.capacity_w-1)*100:.1f}%")


if __name__ == "__main__":
    # Print building summary
    villa = create_saudi_villa()
    total_area = sum(z.area_m2 for z in villa)
    total_cooling = sum(z.hvac.capacity_kw for z in villa)
    total_power = sum(z.hvac.rated_power_kw for z in villa)
    
    print("=" * 60)
    print("MULTI-ZONE SAUDI RESIDENTIAL VILLA")
    print("=" * 60)
    for i, z in enumerate(villa):
        print(f"\n  Zone {i+1}: {z.name}")
        print(f"    Area:     {z.area_m2} m²")
        print(f"    HVAC:     {z.hvac.name}")
        print(f"    Capacity: {z.hvac.capacity_btu:,} BTU/h ({z.hvac.capacity_kw:.2f} kW)")
        print(f"    Power:    {z.hvac.rated_power_kw:.2f} kW")
        print(f"    UA:       {z.envelope_UA_W_K:.1f} W/K")
        print(f"    C:        {z.thermal_capacitance_J_K/1e6:.2f} MJ/K")
    
    print(f"\n  TOTALS:")
    print(f"    Floor area:      {total_area:.0f} m²")
    print(f"    Cooling capacity:{total_cooling:.2f} kW ({total_cooling*3412:.0f} BTU/h)")
    print(f"    Peak power draw: {total_power:.2f} kW")
    print(f"    Zones:           {len(villa)}")
    print(f"    AC units:        2× Small (12,600 BTU) + 2× Medium (18,500 BTU) + 1× Large (21,000 BTU)")
    
    print("\n")
    running_example()
