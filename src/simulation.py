import numpy as np
import time
from config import Configuration
from forces import compute_forces_lca, compute_forces_naive
from output_and_plots import save_xyz, track_comp_time

class Simulation:
    def __init__(self, config: Configuration):
        """Initialize the simulation with a configuration object."""
        # Arguments from the command line
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
        self.minimize = config.minimize
        self.minimization_steps = config.minimization_steps
        self.use_lca = config.use_lca

        # Derive box size and initial lattice
        self.box_size = self.compute_box_size()
        self.positions = self.create_lattice()
        # Initialize velocities randomly and adjust to desired temperature
        self.velocities, self.kinetic_energy = self.initialize_velocities()
        
        # Choose the force computation method for the simulation
        if self.use_lca:
            self.compute_forces = compute_forces_lca
        else:
            self.compute_forces = compute_forces_naive

        # Initialize forces and potential energy using the chosen method
        self.forces, self.potential_energy = self.compute_forces(
            self.positions, self.box_size, self.rcutoff,
            self.sigma, self.epsilon, self.use_pbc
        )

        # Calculate total energy
        self.total_energy = self.kinetic_energy + self.potential_energy
        # .xyz file name based on the dimensions
        self.trajectory_file = f"{config.dimensions}D_trajectory.xyz"

    def compute_box_size(self):
        """Compute box size based on nr of particles and particle density."""
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
        # Generate small random initial velocities
        velocities = np.random.uniform(-0.01, 0.01, size=(self.n_particles, self.dimensions))
        velocities -= np.mean(velocities, axis=0)   # Normalise to have zero net momentum
        
        # Adjust velocities to match the desired system temperature
        kinetic_energy = 0.5 * np.sum(velocities**2) # Kinetic energy with randomly generated velocities
        if self.dimensions == 3:
            desired_kinetic_energy = 0.5 * self.n_particles * 3 * self.temperature # In 3D
        else:
            desired_kinetic_energy = 0.5 * self.n_particles * 2 * self.temperature # In 2D
        scaling = np.sqrt(desired_kinetic_energy / kinetic_energy)
        velocities *= scaling
        return velocities, desired_kinetic_energy

    def velocity_verlet_step(self):
        """Perform one step of Velocity Verlet integration."""
        # Update velocities by half a step assuming particle mass = 1
        self.velocities += 0.5 * self.forces * self.dt
        # Update positions
        temp_positions = self.positions + self.velocities * self.dt
        # Ensure that particles are still in the box
        self.positions = self.apply_boundary_conditions(temp_positions) 
        # Compute new forces
        self.forces, self.potential_energy = self.compute_forces(
            self.positions, self.box_size, self.rcutoff, 
            self.sigma, self.epsilon, self.use_pbc)
        # Update velocities by another half step
        self.velocities += 0.5 * self.forces * self.dt
        # Compute kinetic energy and total energy
        self.kinetic_energy = 0.5 * np.sum(self.velocities ** 2)
        self.total_energy = self.kinetic_energy + self.potential_energy

        return self.kinetic_energy, self.potential_energy, self.total_energy

    def apply_boundary_conditions(self, positions):
        """Apply hard wall or periodic boundary conditions in 2D or 3D."""
        # Use periodic boundary conditions
        if self.use_pbc:
            positions %= self.box_size
        # Use hard walls
        else:
            for i in range(self.n_particles):
                for dim in range(self.dimensions):  # check x, y, and (z) coordinates
                    if positions[i, dim] < 0:
                        positions[i, dim] = -positions[i, dim]
                        self.velocities[i, dim] *= -1  # Flip the impacted velocity
                    elif positions[i, dim] > self.box_size:
                        positions[i, dim] = 2 * self.box_size - positions[i, dim]
                        self.velocities[i, dim] *= -1  # Flip the impacted velocity
        return positions

    def simulate_LJ(self):
        """Run Lennard-Jones simulation."""
        # Intialize timestep and energy lists for data storing
        time_steps = []
        kinetic_energies = []
        potential_energies = []
        total_energies = []

        # Clean/initialize the .xyz trajectory file
        with open(self.trajectory_file, "w") as f:
            pass
        # Save initial positions to .xyz at step 0
        save_xyz(self.positions, self.trajectory_file, 0)

        # Start tracking computational time
        start_time = time.time()

        # Simulation loop
        for step in range(self.steps):
            kinetic_energy, potential_energy, total_energy = self.velocity_verlet_step()

            # Save data for plotting and energy_data.dat
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

        # End tracking computational time
        end_time = time.time()
        # Call track_comp_time to log the computational time data and simulation parameters
        track_comp_time(start_time, end_time, self.steps, self.config)

        # Return collected data
        return time_steps, kinetic_energies, potential_energies, total_energies

    def minimize_energy(self):
        """Minimize the potential energy of the system."""
        # Set velocities to 0
        self.velocities = np.zeros_like(self.positions)

        # Clear .xyz file only if minimizing without running LJ simulation first
        if self.minimize_only:
            with open(self.trajectory_file, "w") as f:
                pass
        # Save positions to .xyz
        save_xyz(self.positions, self.trajectory_file, 0)

        # Compute initial forces with naive or LCA algorithm
        forces, initial_potential_energy = self.compute_forces(self.positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, False)

        # Initialize potential energy and time steps lists
        time_steps = [0]
        potential_energies = [initial_potential_energy]

        # Run the minimization loop
        for step in range(self.minimization_steps):
            # Normalized steepest descent step
            max_force_component = np.max(np.abs(forces))
            normalized_forces = forces / max_force_component
            new_positions = self.positions + self.dt * normalized_forces
            new_positions = self.apply_boundary_conditions(new_positions)

            # Compute new forces and potential energy
            forces, new_potential_energy = self.compute_forces(new_positions, self.box_size, self.rcutoff, self.sigma, self.epsilon, self.use_pbc)

            # Log data for plotting
            time_steps.append(step * self.dt)
            potential_energies.append(new_potential_energy)
            save_xyz(new_positions, self.trajectory_file, step + 1)

            if step % 100 == 0:
                print(f"Step: {step:10d} | Potential Energy: {new_potential_energy:12.9f}")

            # Check for convergence based on energy change
            energy_change = np.abs(new_potential_energy - self.potential_energy)
            energy_threshold = 1e-6
            if energy_change < energy_threshold:
                print(f"Converged due to energy change in {step + 1} steps.")
                return time_steps, potential_energies

            # Update positions and potential energy for the next iteration
            self.positions = new_positions
            self.potential_energy = new_potential_energy

        print(f"Step: {(step+1):10d} | Potential Energy: {new_potential_energy:12.9f}")
        print("Energy minimization did not converge.")
        return time_steps, potential_energies