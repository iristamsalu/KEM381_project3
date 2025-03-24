from numba import jit
import numpy as np

@jit(nopython=True)
def compute_lj_force(r, sigma, epsilon, rcutoff):
    """Compute Lennard-Jones force magnitude with cutoff."""
    if r >= rcutoff or r < 1e-12:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2

    return 24 * epsilon * (2 * inv_r12 - inv_r6) / r

@jit(nopython=True)
def compute_lj_potential(r, sigma, epsilon, rcutoff):
    """Compute shifted Lennard-Jones potential with zero at cutoff."""
    if r >= rcutoff:
        return 0.0
    inv_r = sigma / r
    inv_r6 = inv_r ** 6
    inv_r12 = inv_r6 ** 2
    potential = 4 * epsilon * (inv_r12 - inv_r6)

    # Shift potential to zero at cutoff
    inv_rcut = sigma / rcutoff
    inv_rcut6 = inv_rcut ** 6
    inv_rcut12 = inv_rcut6 ** 2
    shift = 4 * epsilon * (inv_rcut12 - inv_rcut6)

    return potential - shift

@jit(nopython=True)
def compute_forces_naive(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
    """Naive O(N^2) force calculation."""
    n = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if use_pbc:
                rij -= box_size * np.round(rij / box_size)
            r = np.linalg.norm(rij)
            if r < rcutoff:
                fmag = compute_lj_force(r, sigma, epsilon, rcutoff)
                forces[i] += fmag * rij / r
                forces[j] -= fmag * rij / r
                potential_energy += compute_lj_potential(r, sigma, epsilon, rcutoff)

    return forces, potential_energy

def build_linked_cells(positions, box_size, rcutoff):
    """
    Assign particles to cells using the linked-cell algorithm (supports 2D or 3D).

    Parameters:
        positions : (N, D) array of particle positions (D=2 or 3)
        box_size : float, size of the (square or cubic) simulation box
        rcutoff : float, cutoff radius

    Returns:
        head : list of head indices for each cell
        lscl : linked list of particle indices
        lc_dim : list of number of cells in each dimension
        rc : actual cell size (scalar)
    """
    n_particles, dim = positions.shape 

    # Divide box into number of cells per dimension
    # Floor func. to get the largest integer that is less than or equal; func. max to avoid zero division
    lc = max(1, int(np.floor(box_size / rcutoff)))
    # Create a cell with dimensions [lc, lc] or [lc, lc, lc]
    lc_dim = [lc] * dim  

    rc = box_size / lc  # cell size

    # Constant that represents an empty cell/linked list entry
    EMPTY = -1
    # Total number of cells depending on dimension
    if dim == 2:
        lc_xy = lc_dim[0] * lc_dim[1]
        # Initialize the head array with 'empty' values, which means all cells start as unoccupied
        # "head" array will store the index of the most recent particle in each cell
        head = [EMPTY] * lc_xy
    else:
        lc_yz = lc_dim[1] * lc_dim[2]
        lc_xyz = lc_dim[0] * lc_yz
        head = [EMPTY] * lc_xyz
    # Initialize with 'empty' values
    # The lscl array represents a linked list of particles in each cell
    lscl = [EMPTY] * n_particles

    for i in range(n_particles):
        # Determine cell index vector (e.g., [x_idx, y_idx, z_idx]) based on particle's positions
        mc = [int(positions[i][a] / rc) for a in range(dim)]
        # Ensure particle stays inside boundaries
        mc = [min(max(0, idx), lc - 1) for idx in mc]
         # Convert the 2D or 3D cell index vector (mc) to a scalar (1D) index for accessing the 'head' array
        if dim == 2:
            c_index = mc[0] * lc_dim[1] + mc[1]
        else:
            c_index = mc[0] * lc_yz + mc[1] * lc_dim[2] + mc[2]
        # Add the particle to the linked list of particles in the appropriate cell
        # The previous particle (or empty if no particle exists) is now the next particle for particle i
        lscl[i] = head[c_index]
        # Set particle i as the new head of the list in this cell
        head[c_index] = i

    return head, lscl, lc_dim

@jit(nopython=True)
def compute_forces_lca(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
    """
    Compute Lennard-Jones forces and potential energy using the linked-cell algorithm (2D or 3D).

    Parameters:
        positions : (N, D) ndarray of particle positions
        box_size : float (length of cubic or square simulation box)
        rcutoff : float (cutoff radius)
        sigma, epsilon : Lennard-Jones parameters
        use_pbc : bool (whether to apply periodic boundary conditions)

    Returns:
        forces : (N, D) ndarray of forces
        potential_energy : float, total Lennard-Jones potential energy
    """
    n_particles, dim = positions.shape
    head, lscl, lc_dim = build_linked_cells(positions, box_size, rcutoff)

    EMPTY = -1
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    neighbor_offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * dim)).T.reshape(-1, dim)

    for mc in np.ndindex(*lc_dim):
        if dim == 2:
            c_index = mc[0] * lc_dim[1] + mc[1]
        else:
            c_index = mc[0] * lc_dim[1] * lc_dim[2] + mc[1] * lc_dim[2] + mc[2]

        i = head[c_index]
        while i != EMPTY:
            pos_i = positions[i]

            for offset in neighbor_offsets:
                mc1 = np.array(mc) + offset
                rshift = np.zeros(dim)

                valid_cell = True
                for a in range(dim):
                    if use_pbc:
                        if mc1[a] < 0:
                            mc1[a] += lc_dim[a]
                            rshift[a] = -box_size
                        elif mc1[a] >= lc_dim[a]:
                            mc1[a] -= lc_dim[a]
                            rshift[a] = box_size
                    else:
                        if mc1[a] < 0 or mc1[a] >= lc_dim[a]:
                            valid_cell = False
                            break
                if not valid_cell:
                    continue

                if dim == 2:
                    c1 = mc1[0] * lc_dim[1] + mc1[1]
                else:
                    c1 = mc1[0] * lc_dim[1] * lc_dim[2] + mc1[1] * lc_dim[2] + mc1[2]

                j = head[c1]
                while j != EMPTY:
                    if j > i:
                        pos_j = positions[j] + rshift
                        r_ij = pos_i - pos_j
                        dist = np.linalg.norm(r_ij)

                        if dist < rcutoff and dist > 1e-12:
                            f_mag = compute_lj_force(dist, sigma, epsilon, rcutoff)
                            fij = f_mag * (r_ij / dist)

                            forces[i] += fij
                            forces[j] -= fij

                            potential_energy += compute_lj_potential(dist, sigma, epsilon, rcutoff)
                    j = lscl[j]
            i = lscl[i]

    return forces, potential_energy
