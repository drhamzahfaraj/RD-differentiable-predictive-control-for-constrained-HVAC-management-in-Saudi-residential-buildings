"""
Differentiable Multi-Zone Building Model (PyTorch)
Used inside the DPC computational graph for backpropagation.

State vector x ∈ R^12:
  [T1, T2, T3, T4, T5, T_out, I_sol, E_cum, sin(h), cos(h), sin(m), cos(m)]

Action vector u ∈ R^5:
  [u1, u2, u3, u4, u5]  ∈ [0, 1] (duty cycle per zone)
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================
# Building Parameters (from building_model.py)
# ============================================================

ZONE_PARAMS = {
    # (area_m2, UA_W_K, C_J_K_medium, Q_int_W, solar_coeff, Q_hvac_W, P_rated_kW, EER_rated)
    0: (20, 52.4, 1.09e6, 333, 2.43, 3690, 1.15, 3.2),   # Bedroom 1 — Small
    1: (22, 55.8, 1.20e6, 353, 2.67, 3690, 1.15, 3.2),   # Bedroom 2 — Small
    2: (40, 88.2, 2.18e6, 945, 6.08, 6150, 2.12, 2.9),   # Living Room — Large
    3: (30, 73.1, 1.64e6, 810, 3.65, 5420, 1.81, 3.0),   # Kitchen — Medium
    4: (35, 78.5, 1.91e6, 605, 4.26, 5420, 1.81, 3.0),   # Master Bed — Medium
}

N_ZONES = 5
DT = 900.0  # 15-minute timestep (seconds)
INTER_ZONE_UA = 10.0  # W/K

# Adjacency: which zones share walls
ADJACENCY = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3, 4],
    3: [1, 2, 4],
    4: [2, 3],
}

# EER degradation
ALPHA_DEG = 0.018  # K^-1
EER_REF_TEMP = 35.0  # °C

# Comfort
T_MIN = 22.0
T_MAX = 24.0

# SEC tariff
TIER1_RATE = 0.18  # SAR/kWh
TIER2_RATE = 0.30
TIER1_LIMIT = 6000.0  # kWh/month
VAT = 1.15


class DifferentiableBuilding(nn.Module):
    """Differentiable RC building model for DPC backpropagation.
    
    Forward: (x_k, u_k) -> x_{k+1}
    All operations are differentiable w.r.t. u_k.
    """
    
    def __init__(self):
        super().__init__()
        # Register building parameters as buffers (not trainable)
        self.register_buffer('UA', torch.tensor([ZONE_PARAMS[i][1] for i in range(N_ZONES)]))
        self.register_buffer('C', torch.tensor([ZONE_PARAMS[i][2] for i in range(N_ZONES)]))
        self.register_buffer('Q_int', torch.tensor([ZONE_PARAMS[i][3] for i in range(N_ZONES)]))
        self.register_buffer('solar_coeff', torch.tensor([ZONE_PARAMS[i][4] for i in range(N_ZONES)]))
        self.register_buffer('Q_hvac_max', torch.tensor([ZONE_PARAMS[i][5] for i in range(N_ZONES)]))
        self.register_buffer('P_rated', torch.tensor([ZONE_PARAMS[i][6] for i in range(N_ZONES)]))
        self.register_buffer('EER_rated', torch.tensor([ZONE_PARAMS[i][7] for i in range(N_ZONES)]))
        
        # Build adjacency matrix
        adj = torch.zeros(N_ZONES, N_ZONES)
        for i, neighbors in ADJACENCY.items():
            for j in neighbors:
                adj[i, j] = INTER_ZONE_UA
        self.register_buffer('adj_matrix', adj)
    
    def eer_degraded(self, T_out):
        """Temperature-dependent EER: Eq. 2 in paper."""
        degradation = 1.0 - ALPHA_DEG * torch.clamp(T_out - EER_REF_TEMP, min=0.0)
        return self.EER_rated * degradation  # shape: (n_zones,) broadcast with T_out
    
    def forward(self, x, u, occupancy=1.0):
        """Step the building model forward by dt.
        
        Args:
            x: State tensor, shape (batch, 12)
               [T1..T5, T_out, I_sol, E_cum, sin_h, cos_h, sin_m, cos_m]
            u: Control tensor, shape (batch, 5), values in [0, 1]
            occupancy: Scalar occupancy multiplier
            
        Returns:
            x_next: Next state, shape (batch, 12)
        """
        T_zones = x[:, :5]      # (batch, 5)
        T_out = x[:, 5:6]       # (batch, 1)
        I_sol = x[:, 6:7]       # (batch, 1)
        E_cum = x[:, 7:8]       # (batch, 1)
        time_feats = x[:, 8:]   # (batch, 4) — sin/cos hour/month
        
        # Envelope heat transfer: UA_i * (T_out - T_i)
        Q_env = self.UA.unsqueeze(0) * (T_out - T_zones)  # (batch, 5)
        
        # Inter-zone coupling: sum_j UA_ij * (T_j - T_i)
        # T_zones: (batch, 5), adj_matrix: (5, 5)
        Q_adj = torch.matmul(T_zones, self.adj_matrix.T) - T_zones * self.adj_matrix.sum(dim=1).unsqueeze(0)
        
        # Internal gains (scaled by occupancy)
        Q_int = self.Q_int.unsqueeze(0) * occupancy  # (batch, 5)
        
        # Solar gains
        Q_solar = self.solar_coeff.unsqueeze(0) * I_sol  # (batch, 5)
        
        # HVAC cooling (u > 0 means cooling)
        Q_hvac = self.Q_hvac_max.unsqueeze(0) * u  # (batch, 5)
        
        # Energy balance: C * dT/dt = Q_env + Q_adj + Q_int + Q_solar - Q_hvac
        dT = (Q_env + Q_adj + Q_int + Q_solar - Q_hvac) / self.C.unsqueeze(0) * DT
        T_next = T_zones + dT
        
        # Power consumption with degraded EER
        eer = self.eer_degraded(T_out)  # (batch, 1) * (5,) -> broadcast
        P_per_zone = self.P_rated.unsqueeze(0) * u  # Simplified: P = P_rated * duty_cycle
        P_total = P_per_zone.sum(dim=1, keepdim=True)  # (batch, 1) in kW
        
        # Energy this timestep (kWh)
        E_step = P_total * (DT / 3600.0)
        E_cum_next = E_cum + E_step
        
        # Assemble next state (T_out, I_sol, time_feats stay same within rollout step)
        x_next = torch.cat([T_next, T_out, I_sol, E_cum_next, time_feats], dim=1)
        
        return x_next
    
    def compute_power(self, u, T_out):
        """Compute electrical power (kW) for given controls and outdoor temp."""
        P = (self.P_rated.unsqueeze(0) * u).sum(dim=1)
        return P  # (batch,)


class ResidualNetwork(nn.Module):
    """Residual dynamics network: learns Δf = f* - f_nom.
    
    Input: [x_k, u_k] ∈ R^17
    Output: ΔT ∈ R^5 (temperature corrections for 5 zones)
    """
    
    def __init__(self, n_state=12, n_action=5, hidden=64, n_layers=3):
        super().__init__()
        layers = []
        in_dim = n_state + n_action
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, 5))  # Output: 5 zone temperature corrections
        self.net = nn.Sequential(*layers)
        
        # Initialize with small weights (residual should start near zero)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, u):
        """
        Args:
            x: (batch, 12), u: (batch, 5)
        Returns:
            delta_T: (batch, 5) temperature corrections
        """
        inp = torch.cat([x, u], dim=1)
        return self.net(inp)


class CorrectedBuilding(nn.Module):
    """f_hat = f_nom + Δf_φ — the corrected building model."""
    
    def __init__(self, nominal, residual):
        super().__init__()
        self.nominal = nominal
        self.residual = residual
    
    def forward(self, x, u, occupancy=1.0):
        x_nom = self.nominal(x, u, occupancy)
        delta = self.residual(x, u)
        # Apply correction only to temperatures (first 5 dims)
        x_corrected = x_nom.clone()
        x_corrected[:, :5] = x_nom[:, :5] + delta
        return x_corrected


class PolicyNetwork(nn.Module):
    """Neural control policy: π_W(x) -> u.
    
    Input: x ∈ R^12 (building state)
    Output: u ∈ [0, 1]^5 (duty cycle per zone)
    """
    
    def __init__(self, n_state=12, n_action=5, hidden=64, n_layers=4):
        super().__init__()
        layers = []
        in_dim = n_state
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, n_action))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 12) building state
        Returns:
            u: (batch, 5) control actions in [0, 1]
        """
        return torch.sigmoid(self.net(x))


def sec_tariff_cost(P_kw, E_cum_kwh, dt_hours=0.25):
    """Differentiable SEC stepwise tariff cost.
    
    Uses soft approximation of the step function for gradient flow:
    c(E) ≈ tier1 + (tier2 - tier1) * sigmoid((E - limit) / smoothing)
    
    Args:
        P_kw: Power (kW), shape (batch,)
        E_cum_kwh: Cumulative monthly energy (kWh), shape (batch,)
        dt_hours: Timestep in hours
        
    Returns:
        cost: SAR for this timestep, shape (batch,)
    """
    E_step = P_kw * dt_hours
    
    # Soft step function for differentiability
    smoothing = 50.0  # kWh — controls sharpness of transition
    tier_weight = torch.sigmoid((E_cum_kwh - TIER1_LIMIT) / smoothing)
    rate = TIER1_RATE + (TIER2_RATE - TIER1_RATE) * tier_weight
    
    cost = rate * E_step * VAT
    return cost


def comfort_penalty(T_zones, T_min=T_MIN, T_max=T_MAX):
    """Quadratic comfort violation penalty.
    
    Args:
        T_zones: (batch, 5) zone temperatures
    Returns:
        penalty: (batch,) scalar penalty
    """
    over = torch.clamp(T_zones - T_max, min=0.0) ** 2
    under = torch.clamp(T_min - T_zones, min=0.0) ** 2
    return (over + under).sum(dim=1)


def switching_penalty(u_curr, u_prev):
    """Penalise compressor switching (on/off transitions).
    
    Args:
        u_curr, u_prev: (batch, 5)
    Returns:
        penalty: (batch,)
    """
    # Soft switching: penalise large changes in duty cycle
    return ((u_curr - u_prev) ** 2).sum(dim=1)
