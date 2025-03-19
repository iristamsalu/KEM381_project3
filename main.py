from config import parse_args
from simulation import Simulation
from plotting import plot_energy

if __name__ == "__main__":
    # Parse command-line arguments and create a config object
    config = parse_args()
    
    # Initialize the simulation with the configuration object
    sim = Simulation(config)

    if config.minimize_only:
        if config.use_lca:
            print("Performing energy minimization with linked cell algorithm...")
            time_steps, potential_energies = sim.minimize_energy()
            print("Energy minimization with linked cell algorithm is complete.")
            plot_energy(time_steps, potential_energies, "Potential Energy", "minimization_lca.png")
        else:
            print("Performing energy minimization with naive algorithm...")
            time_steps, potential_energies = sim.minimize_energy()
            print("Energy minimization with naive algorithm is complete.")
            plot_energy(time_steps, potential_energies, "Potential Energy", "minimization_naive.png")
    else:
        if config.use_lca:
            print("Performing Lennard-Jones simulation with linked cell algorithm...")
            time_steps, kinetic_energies, potential_energies, total_energies = sim.simulate_LJ()
            print("Lennard-Jones simulation with linked cell algorithm is complete.")
            plot_energy(time_steps, [(kinetic_energies, "Kinetic Energy"), 
                                    (potential_energies, "Potential Energy"), 
                                    (total_energies, "Total Energy")],
                        "Energy", f"lj_{config.dimensions}D_simulation_energy_lca.png", is_multiple=True)
        else:
            print("Performing Lennard-Jones simulation with naive algorithm...")
            time_steps, kinetic_energies, potential_energies, total_energies = sim.simulate_LJ()
            print("Lennard-Jones simulation with naive algorithm is complete.")
            plot_energy(time_steps, [(kinetic_energies, "Kinetic Energy"), 
                                    (potential_energies, "Potential Energy"), 
                                    (total_energies, "Total Energy")],
                        "Energy", f"lj_{config.dimensions}D_simulation_energy_naive.png", is_multiple=True)
