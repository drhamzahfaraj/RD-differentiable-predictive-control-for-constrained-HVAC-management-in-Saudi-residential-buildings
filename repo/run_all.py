"""
Run All Experiments for RD-DPC HVAC Paper
Usage: python run_all.py [--city Riyadh] [--thermal-mass medium]
"""
import sys, os, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hvac', 'simulations'))

from hvac_simulation import simulate_annual, sensitivity_analysis


def main():
    parser = argparse.ArgumentParser(description='RD-DPC HVAC Experiments')
    parser.add_argument('--city', type=str, default='all', choices=['Riyadh', 'Jeddah', 'Abha', 'all'])
    parser.add_argument('--thermal-mass', type=str, default='medium', choices=['low', 'medium', 'high', 'all'])
    parser.add_argument('--sensitivity', action='store_true', help='Run thermal mass sensitivity analysis')
    args = parser.parse_args()

    cities = ['Riyadh', 'Jeddah', 'Abha'] if args.city == 'all' else [args.city]
    results = {}

    for city in cities:
        if args.sensitivity:
            print(f"\n{'='*60}")
            print(f"Sensitivity Analysis: {city}")
            print(f"{'='*60}")
            results[city] = sensitivity_analysis(city)
        else:
            print(f"\n{'='*60}")
            print(f"Annual Simulation: {city} (thermal_mass={args.thermal_mass})")
            print(f"{'='*60}")
            results[city] = simulate_annual(city, thermal_mass=args.thermal_mass)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'hvac', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'hvac_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
