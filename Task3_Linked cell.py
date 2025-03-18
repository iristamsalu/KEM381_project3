import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
from collections import defaultdict


# Define the save_xyz function
def save_xyz(positions, step, filename):
    """Save particle positions to the .xyz file.

    Each particle's position is saved in the format:
    Xi x y z
    where `i` is the particle index (starting from 1), and z = 0.0 for 2D simulations.
    """
    with open(filename, "a") as f:
        pass
        # Write the number of particles
        f.write(f"{len(positions)}\n")
        # Write the step number as a comment line
        f.write(f"Step {step}\n")
        # Write each particle's position with a unique label (X1, X2, X3, ...)
        for i, pos in enumerate(positions, start=1):
            f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # Xi x y z


# Define the time function
def measure_computational_time(n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca,
                               filename):
    """Measure the computational time for the simulation."""
    start_time = time.time()
    simulate(n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, filename, use_lca)
    end_time = time.time()
    return end_time - start_time


def compare_system_size_vs_time(density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca):
    """Compare the system size vs. computational time for both 2D and 3D."""
    system_sizes = [10, 50, 100, 200, 500, 1000]  # Example system sizes
    times_2d = []
    # times_3d = []

    for n_particles in system_sizes:
        time_2d = measure_computational_time(
            n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca, filename="lj_2D_trajectory.xyz"
        )
        times_2d.append(time_2d)

        # For 3D, you need to modify the simulation function to handle 3D cases
        # time_3d = measure_computational_time(
        #     n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca, filename="lj_3D_trajectory.xyz"
        # )
        # times_3d.append(time_3d)

    # Plot the results
    plt.plot(system_sizes, times_2d, label="2D")
    # plt.plot(system_sizes, times_3d, label="3D")
    plt.xlabel("System Size (Number of Particles)")
    plt.ylabel("Computational Time (s)")
    plt.title("System Size vs. Computational Time")
    plt.legend()
    plt.savefig("system_size_vs_time.png")
    plt.clf()  # Clear the figure for the next plot



# Define the parse_args function
def parse_args():
    """Parse and validate the command line arguments."""
    parser = argparse.ArgumentParser(description="Lennard-Jones simulation")
    parser.add_argument("--steps", type=int, default=5000, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.0001, help="Timestep")
    parser.add_argument("--density", type=float, default=0.0006, help="Density of particles")
    parser.add_argument("--n_particles", type=int, default=20, help="Number of particles")
    parser.add_argument("--use_pbc", action="store_true", help="Use periodic boundary conditions (PBC)")
    parser.add_argument("--temperature", type=float, default=298.0, help="Desired temperature (K)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Lennard-Jones sigma parameter")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Lennard-Jones epsilon parameter")
    parser.add_argument("--rcutoff", type=float, default=2.5, help="Lennard-Jones cutoff radius")
    parser.add_argument("--use_lca", action="store_true", help="Use Linked Cell Algorithm (LCA)")
    args = parser.parse_args()

    # Validate the input
    if args.steps <= 0:
        print("Error: Number of steps must be greater than 0.")
        sys.exit(1)
    if args.dt <= 0:
        print("Error: Time step (dt) must be greater than 0.")
        sys.exit(1)
    if args.density <= 0:
        print("Error: Density must be greater than 0.")
        sys.exit(1)
    if args.n_particles <= 0:
        print("Error: Number of particles must be greater than 0.")
        sys.exit(1)
    if args.temperature <= 0:
        print("Error: Temperature must be greater than 0.")
        sys.exit(1)
    if args.sigma <= 0:
        print("Error: Sigma must be greater than 0.")
        sys.exit(1)
    if args.epsilon <= 0:
        print("Error: Epsilon must be greater than 0.")
        sys.exit(1)
    if args.rcutoff <= 0:
        print("Error: Cutoff radius must be greater than 0.")
        sys.exit(1)
    return (
        args.steps,
        args.dt,
        args.density,
        args.n_particles,
        args.use_pbc,
        args.temperature,
        args.sigma,
        args.epsilon,
        args.rcutoff,
        args.use_lca
    )

def initialize_system(
        n_particles, density, desired_temperature=298.0
):
    """Initialize the system, including the box size, particle positions and velocities."""
    box_size = np.sqrt(n_particles / density)  # Box size
    # Initial particle positions
    positions = create_lattice(box_size, n_particles)

    # Random normally distributed initial velocities
    velocities = np.random.uniform(-0.01, 0.01, size=(n_particles, 2))
    # Center of mass velocities for zero momentum
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity
    # Scale velocities according to the desired temperature
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    desired_kinetic_energy = 0.5 * n_particles * 2 * desired_temperature  # 1/2 N dof k_B T (k_B = 1, dof=2)
    scaling_factor = np.sqrt(desired_kinetic_energy / kinetic_energy)
    velocities *= scaling_factor
    return box_size, velocities, positions

def create_lattice(box_size, n_particles):
    """Generate the initial lattice for particles."""
    # Calculate number of particles per side, ensuring the grid is square
    particles_per_side = int(np.ceil(np.sqrt(n_particles)))  # Take the ceiling to handle non-square cases
    spacing = box_size / particles_per_side

    positions = []
    for i in range(particles_per_side):
        for j in range(particles_per_side):
            # Check if we have reached the desired number of particles
            if len(positions) < n_particles:
                # Add random displacement to avoid perfectly symmetric forces
                small_noise = np.random.uniform(-0.05, 0.05, size=2) * spacing
                # Set particle in the center of the grid cell and add the displacement
                position = [(i + 0.5) * spacing + small_noise[0], (j + 0.5) * spacing + small_noise[1]]
                positions.append(position)
    return np.array(positions)  # Keep only the first n_particles

def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Calculate Lennard-Jones potential with cutoff."""
    if r >= rcutoff:
        return 0.0
    sigma_over_r = sigma / r
    sigma_over_r_6 = sigma_over_r ** 6
    sigma_over_r_12 = sigma_over_r_6 ** 2
    # Original LJ potential
    lj_potential = 4.0 * epsilon * (sigma_over_r_12 - sigma_over_r_6)
    return lj_potential

def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute Lennard-Jones force and shift it by Lennard-Jones force at cutoff."""
    if r >= rcutoff:
        return 0.0
    # Compute standard LJ terms
    sr6 = (sigma / r) ** 6
    sr12 = sr6 ** 2
    # Compute LJ force at cutoff
    sr6_rc = (sigma / rcutoff) ** 6
    sr12_rc = sr6_rc ** 2
    lj_force_rc = 24 * epsilon * (2 * sr12_rc - sr6_rc) / rcutoff
    # Compute LJ at actual distance r
    lj_force = 24 * epsilon * (2 * sr12 - sr6) / r
    # Return shifted force
    return lj_force - lj_force_rc

def compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc):
    """Compute forces between all particles using the Lennard-Jones potential."""
    # Initialize force array
    forces = np.zeros_like(positions)
    # Initialize potential energy
    potential_energy = 0.0

    # Calculate cutoff potential lj_potential_rc and force lj_force_rc
    lj_potential_rc = 4.0 * epsilon * ((sigma / rcutoff) ** 12 - (sigma / rcutoff) ** 6)
    lj_force_rc = 24.0 * epsilon * (2.0 * (sigma / rcutoff) ** 12 - (sigma / rcutoff) ** 6) / rcutoff

    # Loop through all pairs of particles
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Calculate distance vector between particles
            r_vec = positions[j] - positions[i]

            # Apply minimum image convention for periodic boundaries
            if use_pbc:
                r_vec -= box_size * np.round(r_vec / box_size)

            # Calculate squared distance
            r_squared = np.dot(r_vec, r_vec)  # This is r^2, more efficient than np.linalg.norm()^2
            r = np.sqrt(r_squared)  # Get the actual distance

            if r > 0 and r < rcutoff:  # Ensure we are not dividing by zero and apply cutoff
                # Original LJ potential and force
                lj_potential = compute_lj_potential(r, sigma, epsilon, rcutoff)
                lj_force = compute_lj_force(r, sigma, epsilon, rcutoff)

                # Force-shifted potential
                potential_energy += lj_potential - lj_potential_rc - (r - rcutoff) * lj_force_rc
                # Force-shifted force
                force_magnitude = lj_force - lj_force_rc
                force_vector = force_magnitude * r_vec / r  # Normalize r_vec to get the force direction

                # Apply Newton's third law: equal and opposite forces
                forces[i] -= force_vector
                forces[j] += force_vector
    return forces, potential_energy

def build_linked_cells(positions, box_size, rcutoff):
    """Create a linked cell list for neighbor searching."""
    cell_size = rcutoff  # Each cell has a size equal to the cutoff radius
    num_cells = int(np.floor(box_size / cell_size))  # Number of cells per side
    cells = defaultdict(list)  # Dictionary to store cell indices and particles

    # Assign particles to cells
    for i, pos in enumerate(positions):
        cell_x = int(np.floor(pos[0] / cell_size))
        cell_y = int(np.floor(pos[1] / cell_size))
        cells[(cell_x, cell_y)].append(i)

    return cells, cell_size, num_cells

def compute_forces_lca(positions, box_size, sigma, epsilon, rcutoff, use_pbc):
    """Compute forces using the Linked Cell Algorithm (LCA)."""
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    # Build linked cells
    cells, cell_size, num_cells = build_linked_cells(positions, box_size, rcutoff)

    # Iterate over all cells
    for (cell_x, cell_y), particles in cells.items():
        # Neighboring cells (including itself)
        neighbors = [
            (cell_x + dx, cell_y + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
        ]

        for p1 in particles:
            for neighbor_cell in neighbors:
                if neighbor_cell in cells:
                    for p2 in cells[neighbor_cell]:
                        if p1 < p2:  # Avoid double counting
                            r_vec = positions[p2] - positions[p1]
                            if use_pbc:
                                r_vec -= box_size * np.round(r_vec / box_size)
                            r_squared = np.dot(r_vec, r_vec)
                            if r_squared < rcutoff ** 2:
                                r = np.sqrt(r_squared)
                                lj_potential = compute_lj_potential(r, sigma, epsilon, rcutoff)
                                lj_force = compute_lj_force(r, sigma, epsilon, rcutoff)
                                potential_energy += lj_potential
                                force_vector = lj_force * r_vec / r
                                forces[p1] -= force_vector
                                forces[p2] += force_vector

    return forces, potential_energy

def run_leapfrog_step(
        positions, velocities, forces, dt,
        box_size, use_pbc, sigma, epsilon, rcutoff
):
    """Update positions and velocities using the Leapfrog algorithm with optional boundary conditions"""
    # Update positions at the full step
    positions += velocities * dt + 0.5 * forces * dt ** 2
    # Apply periodic boundary conditions if enabled
    positions = apply_boundary_conditions(positions, velocities, box_size, use_pbc)

    # Recompute forces at the new positions
    new_forces, potential_energy = compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc)

    # Update velocities at the full step
    velocities += 0.5 * (forces + new_forces) * dt

    # Compute kinetic and total energy
    kinetic_energy, total_energy = compute_kinetic_total_energy(velocities, potential_energy)
    return (
        positions,
        velocities,
        new_forces,
        potential_energy,
        kinetic_energy,
        total_energy,
    )

def apply_boundary_conditions(positions, velocities, box_size, use_pbc):
    """Apply boundary conditions to positions and velocities."""
    if use_pbc:
        # Periodic: wrap positions around
        positions = np.mod(positions, box_size)
    else:
        # Hard wall boundary conditions: reflect off the walls
        mask_low = positions < 0
        positions[mask_low] *= -1  # Reflect off the low side
        velocities[mask_low] = -velocities[mask_low]  # Reflect velocity off the low side

        mask_high = positions > box_size
        positions[mask_high] = 2 * box_size - positions[mask_high]  # Reflect off the high side
        velocities[mask_high] = -velocities[mask_high]  # Reflect velocity off the high side
    return positions

def compute_kinetic_total_energy(velocities, potential_energy):
    """Calculate kinetic energy and total energy (kinetic + potential)."""
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    total_energy = kinetic_energy + potential_energy
    return kinetic_energy, total_energy

def plot(time_steps, energies, label, filename, is_multiple=False):
    """Plot kinetic, potential or total energy to check energy conservation."""
    if is_multiple:
        # Plot multiple energies on the same graph
        for energy, label in energies:
            plt.plot(time_steps, energy, label=label)
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energy Conservation")
        plt.legend()
        plt.xlim(0, max(time_steps))
    else:
        # Single plot for total energy or any individual energy
        plt.plot(time_steps, energies, label=label)
        plt.xlabel("Time")
        plt.ylabel(label)
        plt.title(f"{label} Over Time")
        plt.legend()
        plt.xlim(0, max(time_steps))

    plt.savefig(filename)
    plt.clf()  # Clear the figure for the next plot

# Modify the main simulation function to use LCA
def simulate(
        n_particles, density, dt, steps, use_pbc,
        desired_temperature, sigma, epsilon, rcutoff,
        filename="lj_2D_trajectory.xyz", use_lca=False
):
    """Run the simulation with optional LCA."""
    box_size, velocities, positions = initialize_system(n_particles, density, desired_temperature)

    with open(filename, "w") as f:
        pass
    save_xyz(positions, 0, filename)

    if use_lca:
        compute_forces_func = compute_forces_lca
    else:
        compute_forces_func = compute_forces

    forces, potential_energy = compute_forces_func(positions, box_size, sigma, epsilon, rcutoff, use_pbc)

    time_steps, potential_energies, kinetic_energies, total_energies = [], [], [], []

    for step in range(steps):
        positions, velocities, forces, potential_energy, kinetic_energy, total_energy = run_leapfrog_step(
            positions, velocities, forces, dt, box_size, use_pbc, sigma, epsilon, rcutoff
        )
        save_xyz(positions, step + 1, filename)
        kinetic_energy, total_energy = compute_kinetic_total_energy(velocities, potential_energy)
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)
        time_steps.append(step * dt)

        if step % 100 == 0:
            print(
                f"Step: {step:10d} | "
                f"Total Energy: {total_energy:12.2f} | "
                f"Potential Energy: {potential_energy:12.2f} | "
                f"Kinetic Energy: {kinetic_energy:12.2f}"
            )

    print(f"Trajectory saved to {filename}")
    return kinetic_energies, potential_energies, total_energies, time_steps


if __name__ == "__main__":
    # Parse command-line arguments
    (
        steps, dt, density, n_particles, use_pbc,
        temperature, sigma, epsilon, rcutoff, use_lca
    ) = parse_args()

    # Create a unique folder for each run (Optional)
    run_folder = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_folder, exist_ok=True)

    # Generate unique filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename_xyz = os.path.join(run_folder, f"lj_trajectory_{timestamp}.xyz")
    filename_lca_xyz = os.path.join(run_folder, f"linked_cells_{timestamp}.xyz")
    filename_energy_lca = os.path.join(run_folder, f"lj_2D_all_energies_lca_{timestamp}.png")
    filename_total_energy_lca = os.path.join(run_folder, f"lj_2D_total_energy_lca_{timestamp}.png")
    filename_energy = os.path.join(run_folder, f"lj_2D_all_energies_{timestamp}.png")
    filename_total_energy = os.path.join(run_folder, f"lj_2D_total_energy_{timestamp}.png")

    # Run the comparison of system size vs. computational time
    compare_system_size_vs_time(density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca)

    # Run simulation with LCA
    use_lca = True
    kinetic_energies_lca, potential_energies_lca, total_energies_lca, time_steps_lca = simulate(
        n_particles, density, dt, steps, use_pbc,
        temperature, sigma, epsilon, rcutoff,
        filename=filename_lca_xyz, use_lca=True
    )

    # Run simulation without LCA
    use_lca = False
    kinetic_energies, potential_energies, total_energies, time_steps = simulate(
        n_particles, density, dt, steps, use_pbc,
        temperature, sigma, epsilon, rcutoff,
        filename=filename_xyz, use_lca=False
    )

    # Plot energies for LCA
    energies_to_plot_lca = [
        (kinetic_energies_lca, "LCA Kinetic Energy"),
        (potential_energies_lca, "LCA Potential Energy"),
        (total_energies_lca, "LCA Total Energy")
    ]
    plot(time_steps_lca, energies_to_plot_lca, label="LCA Energy", filename=filename_energy_lca, is_multiple=True)
    plot(time_steps_lca, total_energies_lca, label="LCA Total Energy", filename=filename_total_energy_lca, is_multiple=False)

    # Plot energies without LCA
    energies_to_plot = [
        (kinetic_energies, "Kinetic Energy"),
        (potential_energies, "Potential Energy"),
        (total_energies, "Total Energy")
    ]
    plot(time_steps, energies_to_plot, label="Energy", filename=filename_energy, is_multiple=True)
    plot(time_steps, total_energies, label="Total Energy", filename=filename_total_energy, is_multiple=False)

    print(f"Simulation results saved in: {run_folder}")