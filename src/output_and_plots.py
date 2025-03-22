import matplotlib.pyplot as plt
import os

# Create an output directory if it doesn't exist yet
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_energy(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies, plot_title):
    """Plot energy over time."""
    plt.figure(figsize=(10, 10))
    plt.plot(all_time_steps, all_kinetic_energies, label="Kinetic Energy", linestyle="-", color="b", linewidth=2)
    plt.plot(all_time_steps, all_potential_energies, label="Potential Energy", linestyle="-", color="r", linewidth=2)
    plt.plot(all_time_steps, all_total_energies, label="Total Energy", linestyle="--", color="black", linewidth=2)

    # Plot design details:
    plt.title(plot_title, fontsize=14)
    plt.legend(frameon=True, edgecolor='black', facecolor='white', fontsize=14)
    plt.ylabel("Energy", fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlim(0, max(all_time_steps))
    plt.ylim()
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')

    # Save the plot to output folder
    output_path = os.path.join(OUTPUT_DIR, "energy_plot.png")
    plt.savefig(output_path)

def save_xyz(positions, filename, step):
    """Save particle positions to a .xyz file."""
    # Save to output folder
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"Step {step}\n")
        for i, pos in enumerate(positions, start=1):
            if len(pos) == 2:
                f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # 2D system
            elif len(pos) == 3:
                f.write(f"X{i} {pos[0]} {pos[1]} {pos[2]}\n")  # 3D system

def save_energy_data(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies):
    """Save energy values and timesteps to a .dat file."""
    filename = "energy_data.dat"
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w") as f:
        f.write("# time\tEkin\tEpot\tEtot\n")
        for t, ke, pe, te in zip(all_time_steps, all_kinetic_energies, all_potential_energies, all_total_energies):
            f.write(f"{t:.6f}\t{ke:.6f}\t{pe:.6f}\t{te:.6f}\n")

def track_comp_time(start_time, end_time, steps, config, output_file="computational_times.dat"):
    """Track computational time and append to output file with compact format."""
    total_simulation_time = end_time - start_time
    avg_time_per_step = total_simulation_time / steps

    # Append compact computational time data and simulation parameters to the output file
    with open(os.path.join(OUTPUT_DIR, output_file), "a") as f:
        f.write(f"time: {total_simulation_time:.6f}s, avg/step: {avg_time_per_step:.6f}s, "
                f"dim: {config.dimensions}D, "
                f"N: {config.n_particles}, density: {config.density}, steps: {config.steps}, " 
                f"dt: {config.dt}, PBC: {config.use_pbc}, LCA: {config.use_lca}, "
                f"rcut: {config.rcutoff}, sigma: {config.sigma}, "
                f"eps: {config.epsilon}, temp: {config.temperature}\n")

    # Print a summary of the computational time
    print(f"\nTotal time: {total_simulation_time:.6f} s\nAverage time per step: {avg_time_per_step:.6f} s\n")
    
    return total_simulation_time, avg_time_per_step
