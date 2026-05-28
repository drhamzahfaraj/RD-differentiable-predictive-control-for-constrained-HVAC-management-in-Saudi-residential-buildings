"""
RD-DPC Policy Training v2: Fixes for narrow comfort band
=========================================================
Three key improvements:
1. State normalization (all inputs to [-1, 1])
2. Warm-start from proportional controller (supervised pre-training)
3. Progressive comfort band tightening (20-26 → 21-25 → 22-24°C)
"""

import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'data'))

from models import (DifferentiableBuilding, ResidualNetwork, CorrectedBuilding,
                     PolicyNetwork, sec_tariff_cost, switching_penalty, N_ZONES, DT)
from data_gen import generate_scenarios, build_initial_state, update_exogenous


# ============================================================
# 1. State Normalization
# ============================================================

class StateNormalizer:
    """Normalize state to [-1, 1] for stable training."""
    def __init__(self):
        # [T1..T5, T_out, I_sol, E_cum, sin_h, cos_h, sin_m, cos_m]
        self.means = torch.tensor([23.0]*5 + [30.0, 400.0, 3000.0, 0.0, 0.0, 0.0, 0.0])
        self.stds  = torch.tensor([3.0]*5  + [10.0, 400.0, 3000.0, 1.0, 1.0, 1.0, 1.0])
    
    def normalize(self, x):
        return (x - self.means.to(x.device)) / self.stds.to(x.device)
    
    def denormalize_temp(self, T_norm):
        return T_norm * 3.0 + 23.0


class NormalizedPolicy(nn.Module):
    """Policy with built-in normalization."""
    def __init__(self, normalizer, hidden=64, n_layers=4):
        super().__init__()
        self.normalizer = normalizer
        layers = []
        in_dim = 12
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, N_ZONES))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x_norm = self.normalizer.normalize(x)
        return torch.sigmoid(self.net(x_norm))


# ============================================================
# 2. Warm-Start: Supervised pre-training from proportional controller
# ============================================================

def proportional_controller(T_zones, setpoint=23.0):
    """Hand-tuned proportional controller as teacher."""
    error = T_zones - setpoint
    # Cool proportionally when above setpoint, off when below
    u = torch.clamp(0.4 * error, 0.0, 1.0)
    return u


def warm_start(policy, normalizer, n_samples=10000, epochs=200, lr=1e-3):
    """Pre-train policy to mimic proportional controller."""
    print("  Warm-start: supervised pre-training from proportional controller...")
    
    # Generate random states
    T_zones = torch.rand(n_samples, 5) * 6 + 20  # 20-26°C
    T_out = torch.rand(n_samples, 1) * 20 + 25    # 25-45°C
    I_sol = torch.rand(n_samples, 1) * 800         # 0-800 W/m²
    E_cum = torch.rand(n_samples, 1) * 5000        # 0-5000 kWh
    time_feats = torch.randn(n_samples, 4)         # sin/cos
    
    x = torch.cat([T_zones, T_out, I_sol, E_cum, time_feats], dim=1)
    u_teacher = proportional_controller(T_zones, setpoint=23.0)
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for ep in range(epochs):
        u_pred = policy(x)
        loss = nn.functional.mse_loss(u_pred, u_teacher)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % 50 == 0:
            print(f"    Epoch {ep+1}/{epochs} | MSE: {loss.item():.6f}")
    
    print(f"    Warm-start complete. Final MSE: {loss.item():.6f}")


# ============================================================
# 3. Progressive comfort band + DPC training
# ============================================================

def comfort_penalty_adaptive(T_zones, T_min, T_max):
    """Comfort penalty with adjustable band."""
    over = torch.clamp(T_zones - T_max, min=0.0) ** 2
    under = torch.clamp(T_min - T_zones, min=0.0) ** 2
    return (over + under).sum(dim=1)


def setpoint_tracking(T_zones, setpoint=23.0):
    """Soft tracking loss: penalize deviation from setpoint."""
    return ((T_zones - setpoint) ** 2).mean(dim=1)


def train_policy_v2(corrected_model, policy, city, total_epochs=2000, batch=128, seed=42):
    """Train policy with progressive band tightening."""
    
    scenarios = generate_scenarios(city, 3000, 16, seed=seed)
    optimizer = optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Progressive band schedule: wide → narrow
    # Each phase: (epochs, T_min, T_max, Q_comfort, Q_tracking, Q_cost, Q_switch)
    phases = [
        (400, 20.0, 26.0,  500,  50,  1.0, 0.5),   # Wide band: easy to learn
        (400, 21.0, 25.0, 1000, 100,  2.0, 1.0),   # Tightening
        (400, 21.5, 24.5, 2000, 200,  5.0, 2.0),   # Nearly there
        (400, 22.0, 24.0, 3000, 300,  8.0, 3.0),   # Target band
        (400, 22.0, 24.0, 1000, 100, 15.0, 3.0),   # Cost optimization at target
    ]
    
    best_loss = float('inf')
    best_state = None
    ep_total = 0
    
    for pi, (n_ep, Tmin, Tmax, Qc, Qt, Qco, Qs) in enumerate(phases):
        setpoint = (Tmin + Tmax) / 2
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_ep)
        
        for ep in range(n_ep):
            policy.train()
            idx = torch.randint(0, 3000, (batch,))
            
            x = build_initial_state(
                scenarios['T_init'][idx], scenarios['T_out'][idx, 0],
                scenarios['I_sol'][idx, 0], scenarios['E_cum_init'][idx],
                scenarios['hour'][idx, 0], scenarios['month'][idx, 0])
            
            total_cost = torch.zeros(batch)
            total_comfort = torch.zeros(batch)
            total_tracking = torch.zeros(batch)
            total_switch = torch.zeros(batch)
            u_prev = torch.zeros(batch, N_ZONES)
            
            for k in range(16):
                if k > 0:
                    x = update_exogenous(x, scenarios['T_out'][idx, k],
                                        scenarios['I_sol'][idx, k],
                                        scenarios['hour'][idx, k],
                                        scenarios['month'][idx, k])
                
                u = policy(x)
                x = corrected_model(x, u)
                
                T_zones = x[:, :5]
                P = corrected_model.nominal.compute_power(u, x[:, 5:6])
                
                total_cost += sec_tariff_cost(P, x[:, 7])
                total_comfort += comfort_penalty_adaptive(T_zones, Tmin, Tmax)
                total_tracking += setpoint_tracking(T_zones, setpoint)
                total_switch += switching_penalty(u, u_prev)
                u_prev = u.detach()
            
            loss = (Qco * total_cost + Qc * total_comfort + 
                    Qt * total_tracking + Qs * total_switch).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
            sched.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            
            ep_total += 1
            
            if (ep + 1) % 200 == 0:
                with torch.no_grad():
                    T_f = x[:, :5]
                    in_22_24 = ((T_f >= 22) & (T_f <= 24)).float().mean().item() * 100
                    in_band = ((T_f >= Tmin) & (T_f <= Tmax)).float().mean().item() * 100
                    t_mean = T_f.mean().item()
                    u_mean = u.mean().item()
                
                print(f"  P{pi+1} Ep{ep_total:5d} | Band:[{Tmin}-{Tmax}] | "
                      f"Loss:{loss.item():9.0f} | T:{t_mean:.1f}°C | "
                      f"InBand:{in_band:.0f}% | In22-24:{in_22_24:.0f}% | "
                      f"u:{u_mean:.2f}")
    
    if best_state:
        policy.load_state_dict(best_state)
    
    return best_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate_sampled(corrected_model, policy, building, city, seed=42):
    """Evaluate on 4 representative months, scale to annual."""
    from saudi_climate import generate_hourly_temperature, SOLAR_RADIATION
    
    controllers = {
        'Thermostat': lambda x, Tz: torch.where(Tz > 24.5, torch.ones_like(Tz), 
                                                 torch.where(Tz < 23.5, torch.zeros_like(Tz),
                                                            torch.zeros_like(Tz))),
        'RD-DPC': lambda x, Tz: policy(x),
    }
    
    results = {}
    for ctrl_name, ctrl_fn in controllers.items():
        E_total = 0.0; cost_total = 0.0; comfort_ok = 0; steps = 0
        
        for month in [1, 4, 7, 10]:
            days = [31,28,31,30,31,30,31,31,30,31,30,31][month-1]
            T_hr = generate_hourly_temperature(city, month, days)
            I_peak = SOLAR_RADIATION[city][month-1]
            T_zones = torch.full((1, 5), 23.5, dtype=torch.float32)
            E_month = 0.0
            
            for d in range(days):
                for s in range(0, 96, 2):  # 30-min steps for speed
                    h = (s / 4) % 24
                    hi = min(d * 24 + int(h), len(T_hr) - 1)
                    T_out = float(T_hr[hi])
                    I_sol = float(I_peak * max(0, np.sin(np.pi * (h-6)/12)) if 6 <= h <= 18 else 0)
                    
                    x = torch.tensor([[
                        *T_zones.flatten().tolist(), T_out, I_sol, E_month,
                        np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                        np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12)
                    ]], dtype=torch.float32)
                    
                    with torch.no_grad():
                        u = ctrl_fn(x, T_zones)
                        x_next = corrected_model(x, u)
                        T_zones = x_next[:, :5].clamp(18, 35)
                        
                        P = float((building.P_rated * u).sum().item())
                        E_step = P * 0.5  # 30-min steps
                        E_total += E_step; E_month += E_step
                        
                        rate = 0.18 if E_month <= 6000 else 0.30
                        cost_total += rate * E_step * 1.15
                        
                        T_z = T_zones.numpy().flatten()
                        if all(22.0 <= t <= 24.0 for t in T_z):
                            comfort_ok += 1
                        steps += 1
            E_month = 0.0
        
        # Scale 4 months to annual (×3)
        results[ctrl_name] = {
            'energy_kwh': round(E_total * 3),
            'cost_sar': round(cost_total * 3),
            'comfort_pct': round(comfort_ok / steps * 100, 1),
        }
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs('results', exist_ok=True)
    
    normalizer = StateNormalizer()
    all_results = {}
    
    for city in ['Riyadh', 'Jeddah', 'Abha']:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  {city}: Policy Training v2 (progressive band + warm-start)")
        print(f"{'='*60}")
        
        # Build models
        building = DifferentiableBuilding()
        residual = ResidualNetwork()
        policy = NormalizedPolicy(normalizer)
        corrected = CorrectedBuilding(building, residual)
        
        # Load pre-trained residual
        ckpt = torch.load(f'results/residual_{city.lower()}.pt', weights_only=True)
        residual.load_state_dict(ckpt['residual'])
        for p in residual.parameters():
            p.requires_grad = False
        
        # Warm-start from proportional controller
        warm_start(policy, normalizer, n_samples=10000, epochs=200, lr=1e-3)
        
        # Progressive band training
        best_loss = train_policy_v2(corrected, policy, city, total_epochs=2000, batch=128, seed=42)
        
        # Save
        torch.save({
            'residual': residual.state_dict(),
            'policy': policy.state_dict(),
            'best_loss': best_loss,
        }, f'results/model_v2_{city.lower()}.pt')
        
        # Evaluate
        policy.eval()
        results = evaluate_sampled(corrected, policy, building, city, seed=42)
        
        elapsed = time.time() - t0
        
        Et = results['Thermostat']['energy_kwh']
        Er = results['RD-DPC']['energy_kwh']
        savings = round((1 - Er / Et) * 100, 1) if Et > 0 else 0
        
        print(f"\n  Results for {city}:")
        print(f"    Thermostat: {Et:,} kWh, {results['Thermostat']['cost_sar']:,} SAR, "
              f"{results['Thermostat']['comfort_pct']}% comfort")
        print(f"    RD-DPC:     {Er:,} kWh, {results['RD-DPC']['cost_sar']:,} SAR, "
              f"{results['RD-DPC']['comfort_pct']}% comfort")
        print(f"    Savings: {savings}% | Time: {elapsed:.0f}s")
        
        all_results[city] = {
            'Thermostat': results['Thermostat'],
            'RD-DPC': results['RD-DPC'],
            'savings_pct': savings,
            'training_time_s': round(elapsed),
        }
    
    with open('results/trained_results_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))
    
    print(f"\n{'='*60}")
    print("  TRAINING v2 COMPLETE")
    print(f"{'='*60}")
    for c, r in all_results.items():
        print(f"  {c}: Therm={r['Thermostat']['energy_kwh']:,} | "
              f"RD-DPC={r['RD-DPC']['energy_kwh']:,} | "
              f"Savings={r['savings_pct']}% | "
              f"Comfort={r['RD-DPC']['comfort_pct']}%")


if __name__ == '__main__':
    main()
