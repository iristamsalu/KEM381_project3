import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from dataclasses import dataclass

@dataclass
class configuration:
    steps: int
    dt: float
    density: float
    n_particles: int
    use_pbc: bool
    temperature: float
    sigma: float
    epsilon: float
    rcutoff: float
    minimize_only: bool

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
    return configuration(
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

class simulation:
    def __init__(self, config: configuration):
        """Initialize the simulation with a configuration object."""
        self.config = config
        self.n_particles = config.n_particles
        self.density = config.density
        self.dt = config.dt
        self.steps = config.steps
        self.use_pbc = config.use_pbc
        self.temperature = config.temperature
        self.sigma = config.sigma
        self.epsilon = config.epsilon
        self.rcutoff = config.rcutoff
        self.minimize_only = config.minimize_only

        self.box_size = self.compute_box_size()
        self.positions = self.create_lattice()
        self.velocities = self.initialize_velocities()
        self.forces, self.potential_energy = self.compute_forces()
        self.kinetic_energy = 0.5 * np.sum(self.velocities ** 2)
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.trajectory_file = "trajectory.xyz"

    def compute_box_size(self):
        """Compute box size"""
        return (self.n_particles / self.density) ** (1 / 2)

    def create_lattice(self):
        """Create an initial square lattice of particles with slight random displacements."""
        particles_per_side = int(np.ceil(np.sqrt(self.n_particles)))
        spacing = self.box_size / particles_per_side
        positions = []
        for i in range(particles_per_side):
            for j in range(particles_per_side):
                if len(positions) < self.n_particles:
                    small_noise = np.random.uniform(-0.05, 0.05, size=2) * spacing
                    positions.append([(i + 0.5) * spacing + small_noise[0], (j + 0.5) * spacing + small_noise[1]])
        return np.array(positions)

    def initialize_velocities(self):
        """Generate initial velocities"""
        velocities = np.random.uniform(-0.01, 0.01, size=(self.n_particles, 2))
        velocities -= np.mean(velocities, axis=0)
        # Adjust velocities according to the desired temperature 
        kinetic_energy = 0.5 * np.sum(velocities**2)
        desired_kinetic_energy = 0.5 * self.n_particles * 2 * self.temperature
        scaling = np.sqrt(desired_kinetic_energy / kinetic_energy)
        velocities *= scaling
        return velocities

    def compute_lj_potential(self, r):
        """Calculate the Lennard-Jones potential with cutoff."""
        if r >= self.rcutoff:
            return 0.0
        sr6 = (self.sigma / r)**6
        sr12 = sr6**2
        return 4 * self.epsilon * (sr12 - sr6)

    def compute_lj_force(self, r):
        """Compute the Lennard-Jones force with cutoff."""
        if r >= self.rcutoff:
            return 0.0
        sr6 = (self.sigma / r)**6
        sr12 = sr6**2
        return 24 * self.epsilon * (2 * sr12 - sr6) / r

    def compute_forces(self):
        """Compute forces and potential energy between all particles."""
        forces = np.zeros_like(self.positions)
        potential_energy = 0.0

        for i in range(len(self.positions)):
            for j in range(i + 1, len(self.positions)):
                r_vec = self.positions[j] - self.positions[i]
                if self.use_pbc:
                    r_vec -= self.box_size * np.round(r_vec / self.box_size)

                r = np.linalg.norm(r_vec)
                if 0 < r < self.rcutoff:  # Corrected condition: r > 0
                    lj_potential = self.compute_lj_potential(r)
                    lj_force = self.compute_lj_force(r)
                    potential_energy += lj_potential

                    force_vector = lj_force * r_vec / r
                    forces[i] -= force_vector
                    forces[j] += force_vector
        return forces, potential_energy

    def run_leapfrog_step(self):
        """Update positions and velocities using the Leapfrog algorithm."""
        self.positions += self.velocities * self.dt + 0.5 * self.forces * self.dt**2
        self.positions = self.apply_boundary_conditions(self.positions)

        new_forces, potential_energy = self.compute_forces()
        self.velocities += 0.5 * (self.forces + new_forces) * self.dt
        
        kinetic_energy = 0.5 * np.sum(self.velocities ** 2)
        total_energy = kinetic_energy + potential_energy

        self.forces = new_forces
        return kinetic_energy, potential_energy, total_energy

    def apply_boundary_conditions(self, positions):
        """Apply boundary conditions to particle positions."""
        if self.use_pbc:
            positions = np.mod(positions, self.box_size)
        else:
            # Hard wall: particles bounce off the walls
            positions = np.where(positions < 0, -positions, positions)
            positions = np.where(positions > self.box_size, 2 * self.box_size - positions, positions)

            # Keep repeating until everything is in bounds (in case it overshoots more than once)
            while np.any(positions < 0) or np.any(positions > self.box_size):
                positions = np.where(positions < 0, -positions, positions)
                positions = np.where(positions > self.box_size, 2 * self.box_size - positions, positions)
        return positions

    def save_xyz(self, step):
        """Save particle positions to a .xyz file."""
        with open(self.trajectory_file, "a") as f:
            f.write(f"{len(self.positions)}\n")
            f.write(f"Step {step}\n")
            for i, pos in enumerate(self.positions, start=1):
                f.write(f"X{i} {pos[0]} {pos[1]} 0.0\n")

    def simulate_LJ(self):
        """Run the simulation."""
        # Initialize energy and timestep lists
        kinetic_energies, potential_energies, total_energies = [self.kinetic_energy], [self.potential_energy], [self.total_energy]
        time_steps = [0]

        # Clean the .xyz file and write timestep 0
        with open(self.trajectory_file, "w") as f: pass
        self.save_xyz(0)

        # Loop over leap-frog timesteps
        for step in range(self.steps):
            kinetic_energy, potential_energy, total_energy = self.run_leapfrog_step()
            # Store energies
            kinetic_energies.append(kinetic_energy)
            total_energies.append(total_energy)
            potential_energies.append(potential_energy)
            time_steps.append(step * self.dt)
            # Write to trajectory file
            self.save_xyz(step + 1)

            if step % 100 == 0:
                print(
                    f"Step: {step:10d} | "
                    f"Total Energy: {total_energy:12.2f} | "
                    f"Potential Energy: {potential_energy:12.2f} | "
                    f"Kinetic Energy: {kinetic_energy:12.2f}"
                )
        return time_steps, kinetic_energies, potential_energies, total_energies

    def minimize_energy(self):
        """Minimize the potential energy of the system."""
        # Set velocities to 0
        self.velocities = np.zeros_like(self.positions)
        # Clean up .xyz file
        with open(self.trajectory_file, "w") as f: pass
        # Write initial positions
        self.save_xyz(0)

        # Compute initial forces
        forces, potential_energy = self.compute_forces()
        # Initialize potential energy and time steps lists
        potential_energies, time_steps = [potential_energy], [0]
        # Run the minimization loop
        for step in range(self.steps):
            # Normalized steepest descent step
            max_force_component = np.max(np.abs(forces))
            normalized_forces = forces / max_force_component
            new_positions = self.positions + self.dt * normalized_forces
            new_positions = self.apply_boundary_conditions(new_positions)

            new_forces, new_potential_energy = self.compute_forces()

            time_steps.append(step * self.dt)
            potential_energies.append(new_potential_energy)
            self.save_xyz(step + 1)

            if step % 100 == 0:
                print(f"Step: {step:10d} | " f"Potential Energy: {potential_energy:12.9f} | ")

            # Calculate gradient change and check convergence
            if np.linalg.norm(forces) < 1e-5:  # convergence criterion
                print(f"Converged due to force norm in {step + 1} steps.")
                return time_steps, potential_energies
            
            self.positions = new_positions
            forces = new_forces
            potential_energy = new_potential_energy
        print("Energy minimization did not converge.")
        return time_steps, potential_energies


    def plot(self, time_steps, energies, label, filename, is_multiple=False):
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
        plt.xlim(0, self.steps*self.dt)
        plt.title(f"{label} Over Time")
        plt.savefig(filename)
        plt.clf()

if __name__ == "__main__":
    # Parse command-line arguments and create a config object
    config = parse_args()
    # Initialize the simulation with the configuration object
    sim = simulation(config)

    if config.minimize_only:
        print("Performing energy minimization...")
        # Run energy minimization
        time_steps, potential_energies = sim.minimize_energy()
        print("Energy minimization complete.")
        # Plot the potential energy change
        sim.plot(time_steps, potential_energies, "Potential Energy", "minimization.png")

    else:
        print("Performing Lennard-Jones simulation...")
        # Run Lennard-Jones simulation
        time_steps, kinetic_energies, potential_energies, total_energies = sim.simulate_LJ()
        print("Lennard-Jones simulation complete.")
        # Plot kinetic, potential and total energy
        sim.plot(time_steps, [(kinetic_energies, "Kinetic Energy"), 
                             (potential_energies, "Potential Energy"), 
                             (total_energies, "Total Energy")],
                 "Energy", "lj_2D_simulation_energy.png", is_multiple=True)