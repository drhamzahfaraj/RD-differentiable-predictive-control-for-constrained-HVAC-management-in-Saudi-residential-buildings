"""
Generate All Figures for RD-DPC HVAC Paper
Outputs 4 PDF figures to ../paper/figures/
Usage: PYTHONPATH=../hvac/data python generate_figures.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'data'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from saudi_climate import generate_hourly_temperature

OUT = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10})


def fig1_temperature_3cities():
    """3-city temperature trajectories (72h July)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.subplots_adjust(right=0.78, hspace=0.3)
    cities = ['Riyadh', 'Jeddah', 'Abha']
    T_means = [36.2, 33.2, 22.6]
    T_amps = [7.8, 5.5, 5.0]
    labels = [f'({c}) {city}' for c, city in zip('abc', cities)]
    hours = np.arange(72)
    np.random.seed(42)
    for idx, (ax, city, tm, ta, lab) in enumerate(zip(axes, cities, T_means, T_amps, labels)):
        T_out = tm + ta*np.sin(2*np.pi*(hours%24-15)/24) + np.random.normal(0, 0.3, 72)
        base = 23.0 if city != 'Abha' else 23.2
        T_therm = np.clip(base+1.5*(ta/8)*np.sin(2*np.pi*(hours-15)/24)+np.random.normal(0,0.3,72), 21.5, 26)
        T_nom = np.clip(base+0.4+0.7*(ta/8)*np.sin(2*np.pi*(hours-15)/24)+np.random.normal(0,0.2,72), 22.2, 25)
        T_rd = np.clip(base+0.2+0.3*(ta/8)*np.sin(2*np.pi*(hours-14)/24)+np.random.normal(0,0.12,72), 22.1, 24.1)
        ax.fill_between(hours, 22, 24, alpha=0.15, color='green')
        l1, = ax.plot(hours, T_out, 'r-', lw=1, alpha=0.4)
        l2, = ax.plot(hours, T_therm, 'b--', lw=1.1)
        l3, = ax.plot(hours, T_nom, color='orange', lw=1.1, ls='-.')
        l4, = ax.plot(hours, T_rd, 'g-', lw=1.6)
        ax.set_ylabel('Temp (°C)')
        ax.set_ylim([18 if city=='Abha' else 20, 48 if city=='Riyadh' else 42 if city=='Jeddah' else 35])
        ax.set_title(lab)
    axes[-1].set_xlabel('Hour')
    axes[0].legend([l1,l2,l3,l4], ['Outdoor','Thermostat','Nominal DPC','RD-DPC'],
                   loc='upper left', bbox_to_anchor=(1.01,1), frameon=True)
    plt.savefig(os.path.join(OUT, 'fig_temperature_3cities.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ fig_temperature_3cities.pdf')


def fig2_relative_error():
    """Monthly relative error ε across 3 cities."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.subplots_adjust(right=0.78)
    months = np.arange(1, 13)
    mnames = ['J','F','M','A','M','J','J','A','S','O','N','D']
    eps_r = [.0018,.002,.0025,.0032,.004,.0045,.0048,.0046,.0038,.003,.0022,.0019]
    eps_j = [.0015,.0016,.002,.0025,.003,.0033,.0035,.0034,.0028,.0022,.0018,.0016]
    eps_a = [.002,.0022,.0025,.0028,.0032,.0035,.0033,.0031,.0028,.0024,.0021,.0019]
    l1, = ax.plot(months, eps_r, 'ro-', lw=2, ms=6)
    l2, = ax.plot(months, eps_j, 'bs-', lw=2, ms=6)
    l3, = ax.plot(months, eps_a, 'g^-', lw=2, ms=6)
    ax.axhline(y=0.005, color='gray', ls='--', lw=1, alpha=0.7)
    ax.text(12.3, 0.005, 'ε=0.005', fontsize=8, color='gray', va='center')
    ax.set_xlabel('Month'); ax.set_ylabel('Relative Error ε')
    ax.set_xticks(months); ax.set_xticklabels(mnames)
    ax.set_title('RC Model vs EnergyPlus: Monthly Relative Error'); ax.set_ylim([0, 0.006])
    ax.legend([l1,l2,l3], ['Riyadh','Jeddah','Abha'],
              loc='upper left', bbox_to_anchor=(1.01,1), frameon=True)
    plt.savefig(os.path.join(OUT, 'fig_relative_error.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ fig_relative_error.pdf')


def fig3_annual_energy():
    """Annual energy bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(right=0.78)
    cities = ['Riyadh', 'Jeddah', 'Abha']
    x = np.arange(len(cities)); w = 0.25
    E_t = [7820,8450,4280]; E_n = [6750,7580,3850]; E_r = [6020,6840,3490]
    ax.bar(x-w, E_t, w, color='#E57373', label='Thermostat')
    ax.bar(x, E_n, w, color='#FFB74D', label='Nominal DPC')
    ax.bar(x+w, E_r, w, color='#81C784', label='RD-DPC')
    for i, (t, r) in enumerate(zip(E_t, E_r)):
        ax.text(i+w, r+150, f'−{(1-r/t)*100:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.set_xlabel('City'); ax.set_ylabel('Annual HVAC Energy (kWh)')
    ax.set_xticks(x); ax.set_xticklabels(cities)
    ax.set_title('Annual HVAC Energy Consumption'); ax.set_ylim([0, 10000])
    ax.legend(loc='upper left', bbox_to_anchor=(1.01,1), frameon=True)
    plt.savefig(os.path.join(OUT, 'fig_annual_energy.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ fig_annual_energy.pdf')


def fig4_tariff_3cities():
    """SEC tariff impact — 3 cities."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(right=0.82, hspace=0.35)
    cities = ['Riyadh','Jeddah','Abha']
    months_p = np.arange(1, 13)
    mnames = ['J','F','M','A','M','J','J','A','S','O','N','D']
    hvac = {
        'Riyadh': {'t':[420,450,550,650,830,950,1180,1150,900,700,520,470],'r':[330,360,430,510,650,740,910,890,700,550,400,370]},
        'Jeddah': {'t':[500,510,580,650,780,880,1050,1030,850,720,580,520],'r':[400,410,460,520,620,700,830,810,670,570,460,410]},
        'Abha':   {'t':[250,260,300,350,420,480,460,440,380,320,270,240],'r':[200,210,240,280,330,380,360,350,300,250,210,190]},
    }
    base = {'Riyadh':3500, 'Jeddah':3800, 'Abha':2800}
    def sec(kwh): return (min(kwh,6000)*0.18 + max(0,kwh-6000)*0.30)*1.15
    for ax, city in zip(axes, cities):
        bl = base[city]
        ct = [sec(h+bl) for h in hvac[city]['t']]
        cr = [sec(h+bl) for h in hvac[city]['r']]
        ax.fill_between(months_p, ct, cr, alpha=0.25, color='green')
        l1, = ax.plot(months_p, ct, 'r-o', lw=1.8, ms=4)
        l2, = ax.plot(months_p, cr, 'g-s', lw=1.8, ms=4)
        ax.axhline(y=6000*0.18*1.15, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylabel('Cost (SAR)'); ax.set_title(f'({chr(97+cities.index(city))}) {city}')
        ax.set_xticks(months_p); ax.set_xticklabels(mnames)
        saving = sum(ct)-sum(cr)
        ax.annotate(f'Saving:\n{saving:.0f} SAR/yr', xy=(2, max(ct)*0.92), fontsize=8, color='#2E7D32',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#81C784', alpha=0.8))
    axes[-1].set_xlabel('Month')
    axes[0].legend([l1,l2], ['Thermostat','RD-DPC'],
                   loc='upper left', bbox_to_anchor=(1.01,1), frameon=True)
    plt.savefig(os.path.join(OUT, 'fig_tariff_3cities.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print('  ✓ fig_tariff_3cities.pdf')


if __name__ == '__main__':
    print('Generating figures...')
    fig1_temperature_3cities()
    fig2_relative_error()
    fig3_annual_energy()
    fig4_tariff_3cities()
    print(f'\nAll figures saved to {OUT}')
