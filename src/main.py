from config import parse_args
from simulation import Simulation
from output_and_plots import plot_energy, save_energy_data
import numpy as np

if __name__ == "__main__":
    # Parse command-line arguments and create a config object
    config = parse_args()
    # Initialize the simulation with the configuration object
    sim = Simulation(config)

    # Indicate which dimension (2D or 3D) is being used
    print(f"Running simulation in {config.dimensions}D...")

    # Store data for plotting and output .dat files
    all_time_steps = []
    all_kinetic_energies = []
    all_potential_energies = []
    all_total_energies = []

    # Run the Lennard-Jones simulation if no minimize_only flag
    if not config.minimize_only:
        if config.use_lca:
            # Run with the linked cell algorithm
            print("Performing Lennard-Jones simulation with linked cell algorithm...")
            time_steps, kinetic_energies, potential_energies, total_energies = sim.simulate_LJ()
            print("Lennard-Jones simulation with linked cell algorithm is complete.")
        else:
            print("Performing Lennard-Jones simulation with naive algorithm...")
            # Run with the naive algorithm
            time_steps, kinetic_energies, potential_energies, total_energies = sim.simulate_LJ()
            print("Lennard-Jones simulation with naive algorithm is complete.")

        # Store LJ simulation energies and timesteps
        all_time_steps.extend(time_steps)
        all_kinetic_energies.extend(kinetic_energies)
        all_potential_energies.extend(potential_energies)
        all_total_energies.extend(total_energies)

    # Run energy minimization after LJ simulation or start straight from the initial lattice if minimize_only 
    if config.minimize_only or config.minimize:
        print("Performing energy minimization...")
        time_steps_min, potential_energies_min = sim.minimize_energy()
        # Extend simulation timesteps and energies with minimization timesteps and energies
        if config.minimize_only:
            all_time_steps.extend(time_steps_min)
        else:
            all_time_steps.extend([t + all_time_steps[-1] + config.dt for t in time_steps_min])

        all_kinetic_energies.extend([0] * len(time_steps_min))
        all_potential_energies.extend(potential_energies_min)
        # Here potential energy = total energy
        all_total_energies.extend(potential_energies_min)

    # Plot the evaluation of energies troughout the simulation
    plot_title = f"Dimensions: {config.dimensions}D | " \
                 f"Particles: {config.n_particles} | " \
                 f"Density: {config.density} | " \
                 f"Timesteps: {len(all_time_steps)-1} | " \
                 f"Timestep Length: {config.dt}"
    plot_energy(np.array(all_time_steps).flatten(), all_kinetic_energies, all_potential_energies, all_total_energies, plot_title)
    save_energy_data(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies)