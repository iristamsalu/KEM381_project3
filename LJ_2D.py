import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

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
    parser.add_argument("--minimize_only", action="store_true", help="Only minimize energy")
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
        args.minimize_only
    )

def initialize_system(n_particles, density, desired_temperature):
    """Initialize the system: box size, particle positions, and velocities."""
    box_size = np.sqrt(n_particles / density)
    positions = create_lattice(box_size, n_particles)
    velocities = np.random.uniform(-0.01, 0.01, size=(n_particles, 2))
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity

    kinetic_energy = 0.5 * np.sum(velocities**2)
    desired_kinetic_energy = 0.5 * n_particles * 2 * desired_temperature
    scaling_factor = np.sqrt(desired_kinetic_energy / kinetic_energy)
    velocities *= scaling_factor

    return box_size, velocities, positions

def create_lattice(box_size, n_particles):
    """Create a square lattice of particles with slight random displacements."""
    particles_per_side = int(np.ceil(np.sqrt(n_particles)))
    spacing = box_size / particles_per_side
    positions = []

    for i in range(particles_per_side):
        for j in range(particles_per_side):
            if len(positions) < n_particles:
                small_noise = np.random.uniform(-0.05, 0.05, size=2) * spacing
                positions.append([(i + 0.5) * spacing + small_noise[0], (j + 0.5) * spacing + small_noise[1]])

    return np.array(positions)

def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Calculate the Lennard-Jones potential with cutoff (NO SHIFT)."""
    if r >= rcutoff:
        return 0.0
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)

def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute the Lennard-Jones force with cutoff (NO SHIFT)."""
    if r >= rcutoff:
        return 0.0
    sr6 = (sigma / r)**6
    sr12 = sr6**2
    return 24 * epsilon * (2 * sr12 - sr6) / r

def compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc):
    """Compute forces and potential energy between all particles."""
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r_vec = positions[j] - positions[i]

            if use_pbc:
                r_vec -= box_size * np.round(r_vec / box_size)

            r = np.linalg.norm(r_vec)
            if 0 < r < rcutoff:  # Corrected condition: r > 0
                lj_potential = compute_lj_potential(r, sigma, epsilon, rcutoff)
                lj_force = compute_lj_force(r, sigma, epsilon, rcutoff)

                potential_energy += lj_potential

                force_vector = lj_force * r_vec / r
                forces[i] -= force_vector
                forces[j] += force_vector

    return forces, potential_energy

def run_leapfrog_step(positions, velocities, forces, dt, box_size, use_pbc, sigma, epsilon, rcutoff):
    """Update positions and velocities using the Leapfrog algorithm."""
    positions += velocities * dt + 0.5 * forces * dt**2
    positions = apply_boundary_conditions(positions, box_size, use_pbc)
    new_forces, potential_energy = compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc)
    velocities += 0.5 * (forces + new_forces) * dt
    kinetic_energy, total_energy = compute_kinetic_total_energy(velocities, potential_energy)

    return positions, velocities, new_forces, potential_energy, kinetic_energy, total_energy

def apply_boundary_conditions(positions, box_size, use_pbc):
    """Apply boundary conditions to particle positions."""
    if use_pbc:
        positions = np.mod(positions, box_size)
    else:
        # Hard wall: particles bounce off the walls
        positions = np.where(positions < 0, -positions, positions)
        positions = np.where(positions > box_size, 2 * box_size - positions, positions)

        # Keep repeating until everything is in bounds (in case it overshoots more than once)
        while np.any(positions < 0) or np.any(positions > box_size):
            positions = np.where(positions < 0, -positions, positions)
            positions = np.where(positions > box_size, 2 * box_size - positions, positions)
    return positions

def compute_kinetic_total_energy(velocities, potential_energy):
    """Calculate the kinetic and total energy."""
    kinetic_energy = 0.5 * np.sum(velocities**2)
    total_energy = kinetic_energy + potential_energy
    return kinetic_energy, total_energy

def minimize_system(n_particles, density, dt, filename, max_steps=100000, tolerance=1e-8, desired_temperature=293, sigma=1, epsilon=1, rcutoff=2.5, use_pbc=False):
    """Minimize the potential energy of the system."""
    # Initialize the system
    box_size, velocities, positions = initialize_system(n_particles, density, desired_temperature)
    # Set initial velocities to 0
    velocities = np.zeros_like(positions)
    # Clean up .xyz file
    with open(filename, "w") as f: pass
    # Write initial positions
    save_xyz(positions, 0, filename)

    # Compute initial forces
    forces, potential_energy = compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc)
    # Initialize potential energy and time steps lists
    potential_energies, time_steps = [potential_energy], [0]
    # Run the minimization loop
    for step in range(max_steps):
        new_positions = positions + dt * forces
        new_positions = apply_boundary_conditions(new_positions, box_size, use_pbc)
        new_forces, new_potential_energy = compute_forces(new_positions, box_size, sigma, epsilon, rcutoff, use_pbc)

        time_steps.append(step * dt)
        potential_energies.append(new_potential_energy)
        save_xyz(new_positions, step + 1, filename)

        if step % 100 == 0:
            print(f"Step: {step:10d} | " f"Potential Energy: {potential_energy:12.9f} | ")

        # Calculate gradient change and check convergence
        if np.linalg.norm(forces) < tolerance:
            print(f"Converged due to force norm in {step + 1} steps.")
            return time_steps, potential_energies
        
        positions = new_positions
        forces = new_forces
        potential_energy = new_potential_energy
    print("Energy minimization did not converge.")
    return time_steps, potential_energies

def save_xyz(positions, step, filename):
    """Save particle positions to a .xyz file."""
    with open(filename, "a") as f:
        f.write(f"{len(positions)}\n")
        f.write(f"Step {step}\n")
        for i, pos in enumerate(positions, start=1):
            f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")

def plot(time_steps, energies, label, filename, is_multiple=False):
    """Plot energy over time."""
    plt.figure()
    if is_multiple:
        for energy, label in energies:
            plt.plot(time_steps, energy, label=label)
        plt.legend()
    else:
        plt.plot(time_steps, energies, label=label)
    plt.xlabel("Time")
    plt.ylabel(label)
    plt.title(f"{label} Over Time")
    plt.savefig(filename)
    plt.clf()

def simulate(n_particles, density, dt, steps, use_pbc, desired_temperature, sigma, epsilon, rcutoff, filename, minimize_only):
    """Run the simulation."""
    # Initialize the system
    box_size, velocities, positions = initialize_system(n_particles, density, desired_temperature)
    forces, potential_energy = compute_forces(positions, box_size, sigma, epsilon, rcutoff, use_pbc)
    kinetic_energy, total_energy = compute_kinetic_total_energy(velocities, potential_energy)

    # Initialize energy and timestep lists
    kinetic_energies, potential_energies, total_energies = [kinetic_energy], [potential_energy], [total_energy]
    time_steps = [0]

    # Clean the .xyz file and write timestep 0
    with open(filename, "w") as f: pass
    save_xyz(positions, 0, filename)

    # Loop over leap-frog timesteps
    for step in range(steps):
        positions, velocities, forces, potential_energy, kinetic_energy, total_energy = run_leapfrog_step(
            positions, velocities, forces, dt, box_size, use_pbc, sigma, epsilon, rcutoff
        )
        save_xyz(positions, step + 1, filename)

        # Append energies to their respective lists
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)
        potential_energies.append(potential_energy)
        time_steps.append(step * dt)

        if step % 100 == 0:
            print(
                f"Step: {step:10d} | "
                f"Total Energy: {total_energy:12.2f} | "
                f"Potential Energy: {potential_energy:12.2f} | "
                f"Kinetic Energy: {kinetic_energy:12.2f}"
            )
    return time_steps, kinetic_energies, potential_energies, total_energies

if __name__ == "__main__":
    # Parse and validate command line input
    steps, dt, density, n_particles, use_pbc, temperature, sigma, epsilon, rcutoff, minimize_only = parse_args()

    if minimize_only:
        print("Performing energy minimization...")
        # Run energy minimization
        filename = "minimization_trajectory.xyz"
        time_steps, potential_energies = minimize_system(
            n_particles, density, dt, filename, max_steps=100000, tolerance=1e-8, 
            desired_temperature=293, sigma=1, epsilon=1, rcutoff=2.5, use_pbc=False)
        print("Energy minimization complete.")
        plot(time_steps, potential_energies, "Potential Energy", "minimization_energy.png")

    else:
        print("Performing Lennard-Jones simulation...")
        filename = "lj_trajectory.xyz" if not use_pbc else "lj_trajectory_PBC.xyz"
        # Run Lennard-Jones simulation
        time_steps, kinetic_energies, potential_energies, total_energies = simulate(
            n_particles, density, dt, steps, use_pbc, temperature,
            sigma, epsilon, rcutoff, filename, minimize_only=False
        )
        print("Lennard-Jones simulation complete.")
        plot(time_steps, [(kinetic_energies, "Kinetic Energy"), (potential_energies, "Potential Energy"), (total_energies, "Total Energy")],
             "Energy", "lj_2D_simulation_energy.png", is_multiple=True)
        