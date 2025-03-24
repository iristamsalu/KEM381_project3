from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import time

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
    parser.add_argument("--lca_method", type=str, default="default", choices=["default", "method2"], help="Select LCA method")
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
        args.use_lca,
        args.lca_method
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


def simulate_with_timing(n_particles, density, use_pbc, sigma, epsilon, rcutoff, method="lj"):
    """Run the simulation and measure computation time without calling `compare_performance()`."""
    print(f"Timing simulation for {n_particles} particles using `{method}`")

    start_time = time.time()

    filename = f"lj_2D_trajectory_{n_particles}_{method}.xyz"
    simulate(n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff,
             filename=filename, method=method)

    end_time = time.time()
    return end_time - start_time


def compare_performance():
    """Compare brute force and linked cell computation times."""
    n_list = [10,50,100,500,1000,2000]
    times_lj, times_linked = [], []

    for N in n_list:
        print(f"Running performance comparison for {N} particles...")

        # Brute-force LJ method (O(N^2))
        t_brute = simulate_with_timing(N, density=0.0006, use_pbc=True,
                                       sigma=1, epsilon=1, rcutoff=2.5, method="lj")
        times_lj.append(t_brute)

        # Linked Cell method (O(N))
        t_linked = simulate_with_timing(N, density=0.0006, use_pbc=True,
                                        sigma=1, epsilon=1, rcutoff=2.5, method="linked")
        times_linked.append(t_linked)

    # Plot performance comparison only once
    plt.figure()
    plt.plot(n_list, times_lj, label="Lennard-Jones (O(N^2))", marker="o")
    plt.plot(n_list, times_linked, label="Linked Cell (O(N))", marker="s")
    plt.xlabel("Number of Particles")
    plt.ylabel("Computation Time (s)")
    plt.legend()
    plt.title("Computation Time vs. System Size")
    plt.grid()

    # Save and show the plot only once
    plt.savefig("performance_comparison.png")

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
    """Create a linked cell list for neighbor searching using LC."""
    cell_size = rcutoff   # Cell size equal to cutoff radius
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
        # Consider only relevant neighbors (including the cell itself)
        neighbor_cells = [
            (cell_x + dx, cell_y + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]  # Only consider direct neighbors
        ]

        for p1 in particles:
            for neighbor_cell in neighbor_cells:
                if neighbor_cell in cells:
                    for p2 in cells[neighbor_cell]:
                        if p1 < p2:  # Avoid double counting
                            r_vec = positions[p2] - positions[p1]
                            if use_pbc:
                                r_vec -= box_size * np.round(r_vec / box_size)
                            r_squared = np.dot(r_vec, r_vec)

                            if r_squared < rcutoff ** 2:  # Check within cutoff distance
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
        box_size, use_pbc, sigma, epsilon, rcutoff, method="lj"
):
    """Update positions and velocities using the Leapfrog algorithm with optional boundary conditions."""
    positions += velocities * dt + 0.5 * forces * dt ** 2
    positions = apply_boundary_conditions(positions, velocities, box_size, use_pbc)

    # Select force computation method
    if method == "linked":
        compute_forces_func = compute_forces_lca
    else:
        compute_forces_func = compute_forces

    new_forces, potential_energy = compute_forces_func(positions, box_size, sigma, epsilon, rcutoff, use_pbc)

    velocities += 0.5 * (forces + new_forces) * dt
    kinetic_energy, total_energy = compute_kinetic_total_energy(velocities, potential_energy)

    return positions, velocities, new_forces, potential_energy, kinetic_energy, total_energy


def apply_boundary_conditions(positions, velocities, box_size, use_pbc):
    """Apply boundary conditions to positions and velocities."""
    if use_pbc:
        positions = np.mod(positions, box_size)
    else:
        # Reflect off boundaries using masks
        mask_low = positions < 0
        mask_high = positions > box_size

        # Element-wise reflection
        positions[mask_low] = -positions[mask_low]
        positions[mask_high] = 2 * box_size - positions[mask_high]

        velocities[mask_low] *= -1
        velocities[mask_high] *= -1

    return positions

def compute_kinetic_total_energy(velocities, potential_energy):
    """Calculate kinetic energy and total energy (kinetic + potential)."""
    kinetic_energy = 0.5 * np.sum(velocities ** 2)
    total_energy = kinetic_energy + potential_energy
    return kinetic_energy, total_energy


def save_xyz(positions, step, filename="lj_trajectory.xyz"):
    """Save particle positions to the .xyz file.

    Each particle's position is saved in the format:
    Xi x y z
    where `i` is the particle index (starting from 1), and z = 0.0 for 2D simulations.
    """
    with open(filename, "a") as f:
        # Write the number of particles
        f.write(f"{len(positions)}\n")
        # Write the step number as a comment line
        f.write(f"Step {step}\n")
        # Write each particle's position with a unique label (X1, X2, X3, ...)
        for i, pos in enumerate(positions, start=1):
            f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")  # Xi x y z


def simulate(
        n_particles, density, dt, steps, use_pbc,
        desired_temperature, sigma, epsilon, rcutoff,
        filename="lj_2D_trajectory.xyz", method="lj"
):
    """Run the simulation."""
    # Initialize the system
    box_size, velocities, positions = initialize_system(n_particles, density, desired_temperature)

    if method == "linked":
        compute_forces_func = compute_forces_lca
    else:
        compute_forces_func = compute_forces

    forces, potential_energy = compute_forces_func(positions, box_size, sigma, epsilon, rcutoff, use_pbc)


    # Clear the .xyz file before starting
    with open(filename, "w") as f:
        pass
    # Write initial positions to the .xyz file
    save_xyz(positions, 0, filename)

    # Compute initial forces
    forces, potential_energy = compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc)

    # Lists to store energy values for plotting
    time_steps = []
    potential_energies, kinetic_energies, total_energies = [], [], []

    # Simulation loop
    for step in range(steps):
        # Run leapfrog step
        positions, velocities, forces, potential_energy, kinetic_energy, total_energy = run_leapfrog_step(
            positions=positions,
            velocities=velocities,
            forces=forces,
            dt=dt,
            box_size=box_size,
            use_pbc=use_pbc,
            sigma=sigma,
            epsilon=epsilon,
            rcutoff=rcutoff,
        )
        # Write new coordinates to the .xyz file
        save_xyz(positions, step + 1, filename)

        # Calculate and store kinetic, potential and total energy
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)
        # Store timestep
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


def plot(time_steps, energies, label, filename, is_multiple=False):
    """Plot kinetic, potential, or total energy to check energy conservation."""
    plt.figure()
    if is_multiple:
        for energy_values, energy_label in energies:
            plt.plot(time_steps, energy_values, label=energy_label)
    else:
        plt.plot(time_steps, energies, label=label)

    plt.xlabel("Time")
    plt.ylabel("Energy" if is_multiple else label)
    plt.title("Energy Conservation" if is_multiple else f"{label} Over Time")
    plt.legend()
    plt.xlim(0, max(time_steps))
    plt.savefig(filename)
    plt.clf()  # Clear the figure for the next plot


# Run the simulation with or without PBC
if __name__ == "__main__":
    steps, dt, density, n_particles, use_pbc, temperature, sigma, epsilon, rcutoff, use_lca, lca_method = parse_args()

    if not use_pbc:
        print("Simulating without PBCs...")
    else:
        print("Simulating with PBCs...")

    method = "linked" if use_lca else "lj"

    n_particles_list = [10,50,100,500,1000,2000]

    for n_particles in n_particles_list:
        print(f"Running simulation for {n_particles} particles using method: {method}")

        filename_xyz = f"lj_2D_trajectory_{n_particles}_{method}.xyz"
        filename_energy = f"lj_2D_all_energies_{n_particles}_{method}.png"
        filename_total_energy = f"lj_2D_total_energy_{n_particles}_{method}.png"

        kinetic_energies, potential_energies, total_energies, time_steps = simulate(
            n_particles, density, dt, steps, use_pbc, temperature, sigma, epsilon, rcutoff,
            filename=filename_xyz, method=method
        )

        energies_to_plot = [
            (kinetic_energies, "Kinetic Energy"),
            (potential_energies, "Potential Energy"),
            (total_energies, "Total Energy")
        ]
        plot(time_steps, energies_to_plot, label="Energy", filename=filename_energy, is_multiple=True)
        plot(time_steps, total_energies, label="Total Energy", filename=filename_total_energy, is_multiple=False)

    # Run performance comparison
    compare_performance()
