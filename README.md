# RD-DPC HVAC: Residual-Dynamics Differentiable Predictive Control for Energy-Efficient HVAC Management in Saudi Residential Buildings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](repo/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

**Target Journal:** Applied Energy (Elsevier) — IF 11.0, CAS Zone 1 TOP

---

## Author

**Hamzah Faraj**  
Department of Computer Science, Taif University, Taif 21944, Saudi Arabia  
Email: h.faraj@tu.edu.sa · ORCID: [0009-0009-8832-0407](https://orcid.org/0009-0009-8832-0407)

---

## Abstract

HVAC systems account for over 50% of building energy consumption in Saudi Arabia, where outdoor temperatures exceeding 45°C necessitate near-continuous cooling. This paper proposes **Residual-Dynamics DPC (RD-DPC)**, an intelligent HVAC control framework that learns a neural control policy offline by backpropagating through a corrected building thermal model. Unlike classical approaches assuming linear dynamics (Mousa, Schewe & Wojtczak, 2016), RD-DPC augments the nominal RC thermal model with a feedforward neural network capturing unmodeled nonlinearities, and optimises over the Saudi stepwise electricity tariff (0.18/0.30 SAR/kWh + 15% VAT). Validated on a 5-zone, 147 m² Saudi villa across Riyadh, Jeddah, and Abha, RD-DPC reduces annual HVAC energy by **18–23%** vs thermostat and **8–14%** vs nominal DPC, while maintaining **95–97%** comfort compliance within the SBC 601 range of 22–24°C. Annual cost savings: **780–1,450 SAR/household**. Payback period: **18 months**.

**Keywords:** Intelligent HVAC control · Differentiable predictive control · Residual dynamics · Smart buildings · Energy optimisation · Saudi Building Code · Stepwise tariff

---

## Repository Structure

```
RD-DPC-HVAC/
├── paper/
│   ├── main.tex                    # LaTeX source (Applied Energy format)
│   ├── main.pdf                    # Compiled paper (32 pages)
│   ├── references.bib              # 33 cited references (28 with verified DOIs)
│   └── figures/
│       ├── fig_temperature_3cities.pdf  # 3-city temperature trajectories
│       ├── fig_relative_error.pdf       # Monthly ε vs EnergyPlus
│       ├── fig_annual_energy.pdf        # Annual energy comparison
│       └── fig_tariff_3cities.pdf       # SEC tariff impact, 3 cities
├── hvac/
│   ├── data/
│   │   └── saudi_climate.py        # Annual temp profiles (Riyadh/Jeddah/Abha)
│   ├── simulations/
│   │   ├── building_model.py       # 5-zone RC model + 3 AC sizes + EER degradation
│   │   └── hvac_simulation.py      # Full annual sim: 3 controllers × 3 cities
│   └── results/
│       └── hvac_results.json       # Complete annual results
├── repo/
│   └── LICENSE                     # MIT
└── README.md                       # This file
```

---

## Key Results and Findings

### Main Results (Table 4)

| City | Controller | Energy (kWh/yr) | Cost (SAR/yr) | Comfort (%) | ε (vs EP) | Savings (%) |
|------|-----------|:-:|:-:|:-:|:-:|:-:|
| Riyadh | Thermostat | 7,820 | 1,639 | 78.3 | 0.0041 | — |
| Riyadh | Nominal DPC | 6,750 | 1,374 | 91.2 | 0.0038 | 13.7 |
| **Riyadh** | **RD-DPC** | **6,020** | **1,193** | **96.8** | **0.0019** | **23.0** |
| Jeddah | Thermostat | 8,450 | 1,838 | 82.1 | 0.0039 | — |
| **Jeddah** | **RD-DPC** | **6,840** | **1,388** | **97.2** | **0.0018** | **19.1** |
| Abha | Thermostat | 4,280 | 859 | 71.5 | 0.0043 | — |
| **Abha** | **RD-DPC** | **3,490** | **693** | **95.1** | **0.0020** | **18.5** |

### Techno-Economic Summary

| Metric | Value |
|--------|-------|
| Deployment cost | 680±120 SAR ($181±32 USD) |
| Payback period | 18 months (Riyadh/Jeddah) |
| 10-year ROI | >550% |
| National impact (10% adoption) | 612 GWh saved, 408 MW peak reduction |

---

## Ablation Study (Table 6 — Riyadh)

| Configuration | Energy (kWh/yr) | Comfort (%) | ΔEnergy (%) |
|---------------|:-:|:-:|:-:|
| **RD-DPC (full)** | **6,020** | **96.8** | — |
| w/o DAgger | 6,580 | 93.1 | +9.3 |
| w/o joint fine-tuning | 6,290 | 95.2 | +4.5 |
| w/o residual (Nominal DPC) | 6,750 | 91.2 | +12.1 |
| Black-box (no f_nom) | 6,890 | 89.5 | +14.5 |
| Linear cost (no tiers) | 6,180 | 96.5 | +2.7 |
| Constant EER (no degradation) | 6,240 | 96.7 | +3.7 |
| Thermostat baseline | 7,820 | 78.3 | +29.9 |

### Literature Comparison (Table 8)

| Study | Method | Zones | Cost Model | Savings |
|-------|--------|:-----:|:----------:|:-------:|
| Mousa et al. (2016) | FPTAS | 1 | Linear | Optimal* |
| Drgoňa et al. (2024) | DPC | 1 | Quadratic | 15–30% |
| Nghiem et al. (2023) | PI-MPC | 3 | ToU | 18–22% |
| Bonassi et al. (2024) | NMPC | 2 | Linear | 12–18% |
| **This work** | **RD-DPC** | **5** | **Stepwise** | **18–23%** |

*Optimal within linear dynamics assumption.

---

## Contributions

1. **Residual-corrected thermal model** — Neural network learns RC→EnergyPlus discrepancy while preserving the differentiable computational graph for end-to-end policy optimisation.
2. **Stepwise tariff-aware policy** — SEC two-tier tariff (0.18/0.30 SAR/kWh) embedded in DPC loss with temporal coupling through cumulative monthly consumption. Extends Mousa et al.'s linear cost model.
3. **Temperature-dependent EER** — Models compressor efficiency degradation at high outdoor temperatures (αdeg = 0.018 K⁻¹), enabling the controller to exploit the efficiency gradient via pre-cooling.
4. **Saudi-specific multi-zone validation** — 5-zone villa, 3 AC sizes, 3 climate zones, SBC 601 comfort, actual SEC tariff structure.
5. **Techno-economic analysis** — Deployment cost, payback period, and national-scale impact assessment.

---

## Methodology: Mechanism of Energy Savings

```
OFFLINE (one-time training)
┌──────────────────────────────────────────────────────┐
│ Stage 1: Residual Model Learning                     │
│   RC model + Neural ΔfΦ → Corrected model f̂          │
│   Trained on 5,000 EnergyPlus transitions            │
│                                                      │
│ Stage 2: Policy Optimisation                         │
│   Neural policy πW minimises:                        │
│   J = Σ c(E_cum)·P(k)·Δt + γ·switching_cost          │
│   s.t. 22°C ≤ Ti ≤ 24°C (SBC 601)                    │
│   where P(k) = Q_hvac/EER(T_out) (temp-dependent)    │
│                                                      │
│ Stage 3: DAgger refinement (2-3 iterations)          │
└──────────────────────────────────────────────────────┘

ONLINE (deployment on ESP32)
┌───────────────────────────────────────────────────────┐
│ xk → πW(xk) → uk    (<0.15 ms, no solver needed)      │
│                                                       │
│ Energy savings come from:                             │
│ 1. Pre-cooling when EER is high (morning, T_out<35°C) │
│ 2. Avoiding Tier 2 charges (keep E_cum < 6000 kWh)    │
│ 3. Exploiting thermal mass for peak-load shifting     │
│ 4. Inter-zone coordination                            │
└───────────────────────────────────────────────────────┘
```

---

## Dataset

Simulation-based study using:
- **Building model:** 5-zone RC thermal network (147 m², Saudi residential villa)
- **Climate data:** Synthetic annual profiles from Weather Atlas averages (1991–2020) for Riyadh, Jeddah, Abha
- **HVAC specs:** 3 sizes of T3-rated split AC (12,600 / 18,500 / 21,000 BTU)
- **Tariff:** SEC residential stepwise (0.18/0.30 SAR/kWh + 15% VAT)
- **Validation:** EnergyPlus v23.2 co-simulation with TMY weather files

---

## Running Experiments

### Prerequisites

```bash
pip install numpy matplotlib scipy torch neuromancer
```

### Run annual simulation

```bash
cd hvac/simulations
PYTHONPATH=../data python3 hvac_simulation.py
```

### Run building model example

```bash
cd hvac/simulations
PYTHONPATH=../data python3 building_model.py
```

### Generate figures

```bash
cd hvac/simulations
PYTHONPATH=../data python3 generate_figures.py   # (included in simulation scripts)
```

---

## Limitations

1. RC thermal model validated against EnergyPlus, not measured building data
2. AC units modelled with temperature-dependent EER but without part-load degradation
3. Humidity (latent loads) excluded — justified for arid climates but limits Jeddah accuracy
4. Deterministic occupancy schedule — stochastic occupancy not modelled
5. Simulation-only validation — hardware-in-the-loop testing on actual Saudi buildings needed
6. Single building typology — extension to commercial buildings and different construction types required

---

## License

MIT License — see [LICENSE](repo/LICENSE)

---

## Acknowledgments

The author acknowledges the Deanship of Graduate Studies and Scientific Research, Taif University for funding this work.

**Software:** [NeuroMANCER](https://github.com/pnnl/neuromancer) (PNNL), [PyTorch](https://pytorch.org/), [EnergyPlus](https://energyplus.net/) (DOE)

```
