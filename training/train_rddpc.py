"""
RD-DPC Training Pipeline for Saudi HVAC Control
================================================

Two-stage training:
  Stage 1: Train residual network Δf_φ on (x,u,x_next) transitions
  Stage 2: Train policy π_W by differentiating DPC loss through corrected model

Usage:
  python train_rddpc.py --city Riyadh --epochs 500 --seed 0
  python train_rddpc.py --city all --epochs 500  # Train for all 3 cities
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, os, sys, argparse, time

sys.path.insert(0, os.path.dirname(__file__))
from models import (
    DifferentiableBuilding, ResidualNetwork, CorrectedBuilding, PolicyNetwork,
    sec_tariff_cost, comfort_penalty, switching_penalty,
    N_ZONES, DT, T_MIN, T_MAX
)
from data_gen import generate_scenarios, build_initial_state, update_exogenous


# ============================================================
# Stage 1: Residual Model Training
# ============================================================

def generate_residual_data(building, city, n_transitions=5000, seed=42):
    """Generate (x, u, x_next_true) transitions for residual training.
    
    x_next_true simulates EnergyPlus by adding systematic corrections
    to the nominal RC model output.
    """
    rng = np.random.default_rng(seed)
    scenarios = generate_scenarios(city, n_scenarios=n_transitions, horizon=2, seed=seed)
    
    X, U, X_next_true = [], [], []
    
    with torch.no_grad():
        for i in range(n_transitions):
            T_init = scenarios['T_init'][i:i+1]
            T_out = scenarios['T_out'][i, 0:1]
            I_sol = scenarios['I_sol'][i, 0:1]
            E_cum = scenarios['E_cum_init'][i:i+1]
            hour = scenarios['hour'][i, 0:1]
            month = scenarios['month'][i, 0:1]
            
            x = build_initial_state(T_init, T_out, I_sol, E_cum, hour, month)
            u = torch.rand(1, N_ZONES) * 0.8  # Random controls
            
            # Nominal model prediction
            x_nom = building(x, u)
            
            # "True" (EnergyPlus-like) prediction: add systematic corrections
            # Thermal mass lag: temperatures change more slowly in reality
            thermal_lag = 0.15 * (x[:, 5] - x[:, :5].mean(dim=1, keepdim=True))
            # Infiltration: slight warming effect
            infiltration = 0.08 * torch.clamp(x[:, 5:6] - 24, min=0) / 20
            # Solar redistribution
            solar_effect = 0.05 * x[:, 6:7] / 800 * torch.randn(1, 5) * 0.3
            
            x_true = x_nom.clone()
            x_true[:, :5] = x_nom[:, :5] + thermal_lag + infiltration + solar_effect
            # Add small noise
            x_true[:, :5] += torch.randn(1, 5) * 0.1
            
            X.append(x)
            U.append(u)
            X_next_true.append(x_true)
    
    return torch.cat(X), torch.cat(U), torch.cat(X_next_true)


def train_residual(building, residual_net, city, n_transitions=5000, epochs=200, lr=1e-3, seed=42):
    """Stage 1: Train residual network on transition data."""
    
    print(f"\n{'='*60}")
    print(f"Stage 1: Residual Model Training — {city}")
    print(f"{'='*60}")
    
    X, U, X_next_true = generate_residual_data(building, city, n_transitions, seed)
    
    # Split train/val
    n_train = int(0.8 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    U_train, U_val = U[:n_train], U[n_train:]
    Y_train = X_next_true[:n_train, :5]  # Only temperatures
    Y_val = X_next_true[n_train:, :5]
    
    optimizer = optim.AdamW(residual_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        residual_net.train()
        
        # Forward: nominal + residual
        with torch.no_grad():
            x_nom = building(X_train, U_train)
        delta = residual_net(X_train, U_train)
        x_pred = x_nom[:, :5] + delta
        
        loss = nn.functional.mse_loss(x_pred, Y_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 20 == 0 or epoch == 0:
            residual_net.eval()
            with torch.no_grad():
                x_nom_val = building(X_val, U_val)
                delta_val = residual_net(X_val, U_val)
                x_pred_val = x_nom_val[:, :5] + delta_val
                val_loss = nn.functional.mse_loss(x_pred_val, Y_val).item()
                
                # Also compute nominal-only error for comparison
                nom_error = nn.functional.mse_loss(x_nom_val[:, :5], Y_val).item()
            
            improvement = (1 - val_loss / nom_error) * 100 if nom_error > 0 else 0
            print(f"  Epoch {epoch+1:4d}/{epochs} | Train: {loss.item():.6f} | Val: {val_loss:.6f} | "
                  f"Nom error: {nom_error:.6f} | Improvement: {improvement:.1f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
    
    print(f"  Residual training complete. Best val loss: {best_val_loss:.6f}")
    return best_val_loss


# ============================================================
# Stage 2: Policy Training via DPC
# ============================================================

def train_policy(corrected_model, policy_net, city, epochs=500, lr=1e-3, 
                 horizon=24, batch_size=256, seed=42,
                 Q_comfort=100.0, Q_switch=5.0, Q_cost=1.0):
    """Stage 2: Train neural policy by differentiating through the model."""
    
    print(f"\n{'='*60}")
    print(f"Stage 2: Policy Training via DPC — {city}")
    print(f"{'='*60}")
    
    scenarios = generate_scenarios(city, n_scenarios=batch_size * 10, horizon=horizon, seed=seed)
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(epochs):
        policy_net.train()
        
        # Sample a batch
        idx = torch.randint(0, len(scenarios['T_init']), (batch_size,))
        
        T_init = scenarios['T_init'][idx]
        T_out_seq = scenarios['T_out'][idx]     # (batch, horizon)
        I_sol_seq = scenarios['I_sol'][idx]
        hour_seq = scenarios['hour'][idx]
        month_seq = scenarios['month'][idx]
        occ_seq = scenarios['occupancy'][idx]
        E_cum = scenarios['E_cum_init'][idx]
        
        # Build initial state
        x = build_initial_state(T_init, T_out_seq[:, 0], I_sol_seq[:, 0], E_cum, 
                                hour_seq[:, 0], month_seq[:, 0])
        
        # Rollout through prediction horizon
        total_cost = torch.zeros(batch_size)
        total_comfort = torch.zeros(batch_size)
        total_switch = torch.zeros(batch_size)
        u_prev = torch.zeros(batch_size, N_ZONES)
        
        for k in range(horizon):
            # Update exogenous variables
            if k > 0:
                x = update_exogenous(x, T_out_seq[:, k], I_sol_seq[:, k],
                                     hour_seq[:, k], month_seq[:, k])
            
            # Get control action from policy
            u = policy_net(x)
            
            # Step model
            x = corrected_model(x, u, occupancy=occ_seq[:, k].mean().item())
            
            # Compute costs
            P = corrected_model.nominal.compute_power(u, T_out_seq[:, k:k+1])
            E_cum_curr = x[:, 7]
            
            # Energy cost (stepwise tariff)
            cost_k = sec_tariff_cost(P, E_cum_curr)
            total_cost += cost_k
            
            # Comfort penalty
            T_zones = x[:, :5]
            comfort_k = comfort_penalty(T_zones)
            total_comfort += comfort_k
            
            # Switching penalty
            switch_k = switching_penalty(u, u_prev)
            total_switch += switch_k
            
            u_prev = u.detach()
        
        # Total DPC loss
        loss = (Q_cost * total_cost + Q_comfort * total_comfort + Q_switch * total_switch).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_cost = total_cost.mean().item()
            avg_comfort = total_comfort.mean().item()
            avg_switch = total_switch.mean().item()
            
            # Compute comfort compliance
            with torch.no_grad():
                T_final = x[:, :5]
                in_range = ((T_final >= T_MIN) & (T_final <= T_MAX)).float().mean().item() * 100
            
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {loss_val:.2f} | "
                  f"Cost: {avg_cost:.2f} | Comfort: {avg_comfort:.4f} | "
                  f"Switch: {avg_switch:.4f} | In-range: {in_range:.1f}%")
        
        if loss_val < best_loss:
            best_loss = loss_val
    
    print(f"  Policy training complete. Best loss: {best_loss:.2f}")
    return loss_history


# ============================================================
# Evaluation
# ============================================================

def evaluate_annual(corrected_model, policy_net, thermostat_fn, city, seed=42):
    """Evaluate controllers over full annual simulation."""
    
    print(f"\n{'='*60}")
    print(f"Evaluation: {city} — Annual Simulation")
    print(f"{'='*60}")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'data'))
    from saudi_climate import generate_hourly_temperature, SOLAR_RADIATION
    
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    steps_per_hour = 4  # 15-min steps
    
    controllers = {
        'Thermostat': thermostat_fn,
        'RD-DPC': lambda x: policy_net(x),
    }
    
    results = {}
    
    for ctrl_name, ctrl_fn in controllers.items():
        total_energy = 0.0
        total_cost = 0.0
        comfort_ok = 0
        total_steps = 0
        all_temps = []
        
        for m_idx in range(12):
            month = m_idx + 1
            n_days = days_per_month[m_idx]
            I_peak = SOLAR_RADIATION[city][m_idx]
            T_hourly = generate_hourly_temperature(city, month, n_days)
            
            T_zones = torch.full((1, 5), 24.0)
            E_cum_month = torch.zeros(1, 1)
            
            for day in range(n_days):
                for step in range(24 * steps_per_hour):
                    hour_of_day = (step / steps_per_hour) % 24
                    hour_idx = min(day * 24 + int(hour_of_day), len(T_hourly) - 1)
                    T_out = T_hourly[hour_idx]
                    I_sol = I_peak * max(0, np.sin(np.pi * (hour_of_day - 6) / 12)) if 6 <= hour_of_day <= 18 else 0
                    
                    # Build state
                    x = torch.cat([
                        T_zones,
                        torch.tensor([[T_out, I_sol]], dtype=torch.float32),
                        E_cum_month,
                        torch.tensor([[
                            np.sin(2*np.pi*hour_of_day/24),
                            np.cos(2*np.pi*hour_of_day/24),
                            np.sin(2*np.pi*month/12),
                            np.cos(2*np.pi*month/12),
                        ]], dtype=torch.float32)
                    ], dim=1)
                    
                    with torch.no_grad():
                        if ctrl_name == 'Thermostat':
                            u = ctrl_fn(T_zones)
                        else:
                            u = ctrl_fn(x)
                        
                        x_next = corrected_model(x, u)
                        T_zones = x_next[:, :5]
                        
                        # Power and energy
                        P = corrected_model.nominal.compute_power(u, torch.tensor([[T_out]], dtype=torch.float32)).item()
                        E_step = P * (DT / 3600)
                        total_energy += E_step
                        E_cum_month += E_step
                        
                        # Cost
                        E_cum_val = E_cum_month.item()
                        rate = 0.18 if E_cum_val <= 6000 else 0.30
                        total_cost += rate * E_step * 1.15
                        
                        # Comfort
                        T_z = T_zones.numpy().flatten()
                        if all(T_MIN <= t <= T_MAX for t in T_z):
                            comfort_ok += 1
                        total_steps += 1
                        
                        all_temps.append(T_z.copy())
            
            E_cum_month = torch.zeros(1, 1)  # Reset monthly
        
        comfort_pct = comfort_ok / total_steps * 100
        T_arr = np.array(all_temps)
        
        results[ctrl_name] = {
            'annual_energy_kwh': round(total_energy, 1),
            'annual_cost_sar': round(total_cost, 1),
            'comfort_pct': round(comfort_pct, 2),
            'T_mean': round(T_arr.mean(), 2),
            'T_max': round(T_arr.max(), 2),
            'T_min': round(T_arr.min(), 2),
        }
        
        print(f"  {ctrl_name}:")
        print(f"    Energy: {total_energy:,.0f} kWh/yr")
        print(f"    Cost:   {total_cost:,.0f} SAR/yr")
        print(f"    Comfort: {comfort_pct:.1f}%")
        print(f"    T_mean: {T_arr.mean():.1f}°C")
    
    # Savings
    if 'Thermostat' in results and 'RD-DPC' in results:
        E_t = results['Thermostat']['annual_energy_kwh']
        E_r = results['RD-DPC']['annual_energy_kwh']
        savings = (1 - E_r / E_t) * 100 if E_t > 0 else 0
        results['savings_pct'] = round(savings, 2)
        print(f"\n  Savings: {savings:.1f}% ({E_t - E_r:,.0f} kWh/yr)")
    
    return results


def thermostat_controller(T_zones, setpoint=24.0, deadband=1.0):
    """Simple on/off thermostat."""
    u = torch.zeros_like(T_zones)
    u[T_zones > setpoint + deadband / 2] = 1.0
    return u


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='RD-DPC HVAC Training')
    parser.add_argument('--city', type=str, default='Riyadh', choices=['Riyadh', 'Jeddah', 'Abha', 'all'])
    parser.add_argument('--epochs-residual', type=int, default=200)
    parser.add_argument('--epochs-policy', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=24, help='Prediction horizon (15-min steps)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    cities = ['Riyadh', 'Jeddah', 'Abha'] if args.city == 'all' else [args.city]
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    for city in cities:
        t_start = time.time()
        
        # Build models
        building = DifferentiableBuilding()
        residual_net = ResidualNetwork()
        policy_net = PolicyNetwork()
        corrected_model = CorrectedBuilding(building, residual_net)
        
        if not args.eval_only:
            # Stage 1: Train residual
            train_residual(building, residual_net, city,
                          n_transitions=5000, epochs=args.epochs_residual,
                          lr=args.lr, seed=args.seed)
            
            # Stage 2: Train policy
            loss_hist = train_policy(corrected_model, policy_net, city,
                                    epochs=args.epochs_policy, lr=args.lr,
                                    horizon=args.horizon, batch_size=args.batch_size,
                                    seed=args.seed)
            
            # Save models
            torch.save({
                'residual': residual_net.state_dict(),
                'policy': policy_net.state_dict(),
                'loss_history': loss_hist,
            }, os.path.join(args.output_dir, f'model_{city.lower()}.pt'))
            print(f"\n  Models saved to {args.output_dir}/model_{city.lower()}.pt")
        else:
            # Load pre-trained models
            ckpt = torch.load(os.path.join(args.output_dir, f'model_{city.lower()}.pt'))
            residual_net.load_state_dict(ckpt['residual'])
            policy_net.load_state_dict(ckpt['policy'])
        
        # Evaluate
        policy_net.eval()
        residual_net.eval()
        results = evaluate_annual(corrected_model, policy_net, thermostat_controller, city, seed=args.seed)
        
        elapsed = time.time() - t_start
        results['training_time_s'] = round(elapsed, 1)
        all_results[city] = results
        
        print(f"\n  Total time for {city}: {elapsed:.1f}s")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'trained_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))
    print(f"\n{'='*60}")
    print(f"All results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
