import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def plot_energy(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies, plot_title, output_dir="output"):
    """Plot energy over time and save the figure to the specified output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 10))
    if not (len(all_time_steps) == len(all_kinetic_energies) == len(all_potential_energies) == len(all_total_energies)):
        raise ValueError("Timestep and energy arrays must have the same length.")

    plt.plot(all_time_steps, all_kinetic_energies, label="Kinetic Energy", linestyle="-", color="b", linewidth=2)
    plt.plot(all_time_steps, all_potential_energies, label="Potential Energy", linestyle="-", color="r", linewidth=2)
    plt.plot(all_time_steps, all_total_energies, label="Total Energy", linestyle="--", color="black", linewidth=2)

    # Plot design
    plt.title(plot_title, fontsize=14)
    plt.legend(frameon=True, edgecolor='black', facecolor='white', fontsize=14)
    plt.ylabel("Energy", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlim(0, max(all_time_steps))
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')

    # Save the plot
    output_path = os.path.join(output_dir, "energy_plot.png")
    plt.savefig(output_path)

def save_xyz(positions, filename, step):
    """Save particle positions to a .xyz file in the current working directory."""    
    # Open in "a" mode to append data
    with open(filename, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"Step {step}\n")
        for i, pos in enumerate(positions, start=1):
            if len(pos) == 2:
                f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # 2D system
            elif len(pos) == 3:
                f.write(f"X{i} {pos[0]} {pos[1]} {pos[2]}\n")  # 3D system

def save_energy_data(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies, output_dir="output"):
    """Save energy values and timesteps to a .dat file in the specified output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Flatten inputs
    all_time_steps = np.ravel(all_time_steps)
    all_kinetic_energies = np.ravel(all_kinetic_energies)
    all_potential_energies = np.ravel(all_potential_energies)
    all_total_energies = np.ravel(all_total_energies)

    if not (len(all_time_steps) == len(all_kinetic_energies) == len(all_potential_energies) == len(all_total_energies)):
        raise ValueError("All input arrays must have the same length.")

    # Define output path
    filename = "energy_data.dat"
    output_path = os.path.join(output_dir, filename)

    # Write data
    with open(output_path, "w") as f:
        f.write("# time\tEkin\tEpot\tEtot\n")
        for t, ke, pe, te in zip(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies):
            f.write(f"{t:.6f}\t{ke:.6f}\t{pe:.6f}\t{te:.6f}\n")

def track_comp_time(start_time, end_time, steps, config, output_file="computational_times.dat"):
    """Track computational time and append to output file in the specified output directory."""
    total_simulation_time = end_time - start_time
    avg_time_per_step = total_simulation_time / steps

    # Get the current date and time
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure the output directory exists
    output_dir = os.path.join("output")  # Set default directory to "output"
    os.makedirs(output_dir, exist_ok=True)

    # Build the full output path
    output_path = os.path.join(output_dir, output_file)

    # Append to the file
    with open(output_path, "a") as f:
        f.write(f"{date}, "
                f"time: {total_simulation_time:.6f}s, avg/step: {avg_time_per_step:.6f}s, "
                f"dim: {config.dimensions}D, "
                f"N: {config.n_particles}, density: {config.density}, steps: {config.steps}, " 
                f"dt: {config.dt}, PBC: {config.use_pbc}, LCA: {config.use_lca}, JIT: {config.use_jit}, "
                f"rcut: {config.rcutoff}, sigma: {config.sigma}, "
                f"eps: {config.epsilon}, temp: {config.temperature}\n")

    # Print a summary of the computational time
    print(f"\nTotal time: {total_simulation_time:.6f} s\nAverage time per step: {avg_time_per_step:.6f} s\n")
    
    return total_simulation_time, avg_time_per_step
