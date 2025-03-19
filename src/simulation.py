import numpy as np
import time
from config import Configuration
from forces import compute_forces_lca, compute_forces_naive
from plotting import save_xyz

class Simulation:
    def __init__(self, config: Configuration):
        """Initialize the simulation with a configuration object."""
        self.config = config
        self.dimensions = config.dimensions
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
        self.use_lca = config.use_lca

        self.box_size = self.compute_box_size()
        self.positions = self.create_lattice()
        self.velocities = self.initialize_velocities()
        
        # Initialize forces and potential energy
        if self.use_lca:
            self.forces, self.potential_energy = compute_forces_lca(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)
        else:
            self.forces, self.potential_energy = compute_forces_naive(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)

        self.kinetic_energy = 0.5 * np.sum(self.velocities ** 2)
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.trajectory_file = f"{config.dimensions}D_trajectory.xyz"

    def compute_box_size(self):
        """Compute box size."""
        if self.dimensions == 3:
            return (self.n_particles / self.density) ** (1 / 3)  # Box size in 3D
        else:
            return (self.n_particles / self.density) ** (1 / 2)  # Box size in 2D

    def create_lattice(self):
        """Create a lattice of particles (2D or 3D) with slight random displacements."""
        # Determine the number of particles per side based on the dimensions
        n_side = int(np.ceil(self.n_particles ** (1 / self.dimensions)))
        spacing = self.box_size / n_side
        positions = []
        for indices in np.ndindex(*([n_side] * self.dimensions)):  # Iterate over grid indices
            if len(positions) < self.n_particles:
                # Create the position based on the current dimension
                position = [(i + 0.5) * spacing for i in indices]
                # Add slight random noise considering dimensions
                noise = np.random.uniform(-0.05, 0.05, size=self.dimensions) * spacing
                position = np.array(position) + noise
                positions.append(position)
        return np.array(positions)

    def initialize_velocities(self):
        """Generate initial velocities."""
        velocities = np.random.uniform(-0.01, 0.01, size=(self.n_particles, self.dimensions))
        velocities -= np.mean(velocities, axis=0)   # Zero net momentum
        kinetic_energy = 0.5 * np.sum(velocities**2)
        # Adjust kinetic energy considering desired system temperature
        if self.dimensions == 3:
            desired_kinetic_energy = 0.5 * self.n_particles * 3 * self.temperature 
        else:
            desired_kinetic_energy = 0.5 * self.n_particles * 2 * self.temperature        
        scaling = np.sqrt(desired_kinetic_energy / kinetic_energy)
        velocities *= scaling
        return velocities

    def velocity_verlet_step(self):
        """Perform one step of Velocity Verlet integration."""
        # 1. Update velocities by half a step
        self.velocities += 0.5 * self.forces * self.dt

        # 2. Update positions
        self.positions += self.velocities * self.dt
        self.positions = self.apply_boundary_conditions(self.positions)

        # 3. Compute new forces
        if self.use_lca:
            new_forces, self.potential_energy = compute_forces_lca(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)
        else:
            new_forces, self.potential_energy = compute_forces_naive(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)

        # 4. Update velocities by another half step
        self.velocities += 0.5 * new_forces * self.dt

        # Compute kinetic energy and total energy
        kinetic_energy = 0.5 * np.sum(self.velocities ** 2)
        total_energy = kinetic_energy + self.potential_energy

        # Update forces
        self.forces = new_forces # Correctly update forces
        return kinetic_energy, self.potential_energy, total_energy

    def apply_boundary_conditions(self, positions):
        """Apply hard wall or periodic boundary conditions in 2D or 3D."""
        # Use periodic boundary conditions
        if self.use_pbc:
            self.positions %= self.box_size
        # Use hard walls
        else:
            for i in range(self.n_particles):
                for dim in range(self.dimensions):  # x, y, and (z)
                    if positions[i, dim] < 0:
                        positions[i, dim] = -positions[i, dim]
                        self.velocities[i, dim] *= -1  # Flip the impacted velocity
                    elif positions[i, dim] > self.box_size:
                        positions[i, dim] = 2 * self.box_size - positions[i, dim]
                        self.velocities[i, dim] *= -1  # Flip the impacted velocity
        return positions

    def simulate_LJ(self):
        """Run the full Lennard-Jones simulation."""
        time_steps = []
        kinetic_energies = []
        potential_energies = []
        total_energies = []

        # Save initial positions to .xyz at step 0
        save_xyz(self.positions, self.trajectory_file, 0)

        # Simulation loop
        for step in range(self.steps):
            kinetic_energy, potential_energy, total_energy = self.velocity_verlet_step()

            # Save data for plotting
            time_steps.append(step * self.dt)
            kinetic_energies.append(kinetic_energy)
            potential_energies.append(potential_energy)
            total_energies.append(total_energy)

            # Save positions to .xyz file at each step
            save_xyz(self.positions, self.trajectory_file, step + 1)

            # Print some progress
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
        save_xyz(self.positions, self.trajectory_file, 0)

        # Compute initial forces with naive or LCA algorithm
        if self.use_lca:
            forces, initial_potential_energy = compute_forces_lca(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)
        else:
            forces, initial_potential_energy = compute_forces_naive(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)

        # Initialize potential energy and time steps lists
        time_steps = [0]
        potential_energies = [initial_potential_energy]

        # Run the minimization loop
        for step in range(self.steps):
            # Normalized steepest descent step
            max_force_component = np.max(np.abs(forces))
            normalized_forces = forces / max_force_component
            new_positions = self.positions + self.dt * normalized_forces
            new_positions = self.apply_boundary_conditions(new_positions)
            # Compute new forces based on new positions
            if self.use_lca:
                forces, new_potential_energy = compute_forces_lca(new_positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)
            else:
                forces, new_potential_energy = compute_forces_naive(new_positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)

            # Save data for plotting
            time_steps.append(step * self.dt)
            potential_energies.append(new_potential_energy)
            save_xyz(new_positions, self.trajectory_file, step + 1)

            if step % 100 == 0:
                print(f"Step: {step:10d} | Potential Energy: {new_potential_energy:12.9f}")

            # Check for convergence based on force and energy
            force_norm = np.linalg.norm(forces)
            energy_change = np.abs(new_potential_energy - self.potential_energy)
            # Define convergence thresholds
            force_threshold = 1e-6
            energy_threshold = 1e-6
            # Convergence check
            if force_norm < force_threshold and energy_change < energy_threshold:
                print(f"Converged due to force norm and energy change in {step + 1} steps.")
                return time_steps, potential_energies

            # Update positions for the next iteration
            self.positions = new_positions
            self.potential_energy = new_potential_energy  # Update the potential energy

        print("Energy minimization did not converge.")
        return time_steps, potential_energies
