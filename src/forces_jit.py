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
def compute_forces_naive_jit(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
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

@jit(nopython=True)
def build_linked_cells(positions, box_size, rcutoff):
    """
    Assign particles to cells using the linked-cell algorithm (supports 2D or 3D).
    """
    n_particles, dim = positions.shape

    # Divide box into number of cells per dimension
    lc = max(1, int(np.floor(box_size / rcutoff)))
    lc_dim = [lc] * dim
    lc_dim = np.array(lc_dim, dtype=np.int64)
    rc = box_size / lc  # cell size

    EMPTY = -1

    if dim == 2:
        lc_xy = lc_dim[0] * lc_dim[1]
        head = np.full(lc_xy, EMPTY, dtype=np.int64)
    else:
        lc_yz = lc_dim[1] * lc_dim[2]
        lc_xyz = lc_dim[0] * lc_yz
        head = np.full(lc_xyz, EMPTY, dtype=np.int64)
    lscl = np.full(n_particles, EMPTY, dtype=np.int64)

    for i in range(n_particles):
        # Determine cell index vector (e.g., [x_idx, y_idx, z_idx]) based on particle's positions
        mc = (positions[i] / rc).astype(np.int64)
        mc = np.minimum(np.maximum(0, mc), lc_dim - 1)

        # Convert the 2D or 3D cell index vector (mc) to a scalar (1D) index for accessing the 'head' array
        if dim == 2:
            c_index = mc[0] * lc_dim[1] + mc[1]
        else:
            c_index = mc[0] * lc_dim[1] * lc_dim[2] + mc[1] * lc_dim[2] + mc[2]

        # Add the particle to the linked list of particles in the appropriate cell
        lscl[i] = head[c_index]
        head[c_index] = i

    return head, lscl, lc_dim

@jit(nopython=True)
def compute_forces_lca_jit(positions, box_size, rcutoff, sigma, epsilon, use_pbc):
    """
    Compute Lennard-Jones forces and potential energy using the linked-cell algorithm (2D or 3D).
    """
    _, dim = positions.shape
    head, lscl, lc_dim = build_linked_cells(positions, box_size, rcutoff)

    forces = np.zeros_like(positions)
    potential_energy = 0.0
    EMPTY = -1

    # Neighbor offsets for both 2D and 3D cases
    if dim == 2:
        neighbor_offsets = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1],
                                     [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    else:
        neighbor_offsets = np.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], 
                                     [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], 
                                     [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], 
                                     [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], 
                                     [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], 
                                     [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], 
                                     [1, 1, -1], [1, 1, 0], [1, 1, 1]], dtype=np.int64)

    # Iterate over cells
    if dim == 2:
        for i in range(lc_dim[0]):
            for j in range(lc_dim[1]):
                mc = np.array([i, j])  # Current cell index
                c_index = mc[0] * lc_dim[1] + mc[1]

                i_particle = head[c_index]
                while i_particle != EMPTY:
                    pos_i = positions[i_particle]
                    # Iterate through neighbor offsets
                    for offset in neighbor_offsets:
                        mc1 = mc + offset
                        rshift = np.zeros(dim)
                        valid_cell = True

                        # Apply periodic boundary conditions (PBC)
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

                        # Compute the linear index for the neighboring cell
                        c1 = mc1[0] * lc_dim[1] + mc1[1]

                        j_particle = head[c1]
                        while j_particle != EMPTY:
                            if j_particle > i_particle:
                                pos_j = positions[j_particle] + rshift
                                r_ij = pos_i - pos_j
                                dist = np.linalg.norm(r_ij)

                                if dist < rcutoff and dist > 1e-12:
                                    f_mag = compute_lj_force(dist, sigma, epsilon, rcutoff)
                                    fij = f_mag * (r_ij / dist)

                                    forces[i_particle] += fij
                                    forces[j_particle] -= fij

                                    potential_energy += compute_lj_potential(dist, sigma, epsilon, rcutoff)
                            j_particle = lscl[j_particle]
                    i_particle = lscl[i_particle]
    else: #dim == 3
        for i in range(lc_dim[0]):
            for j in range(lc_dim[1]):
                for k in range(lc_dim[2]):
                    mc = np.array([i, j, k])  # Current cell index
                    c_index = mc[0] * lc_dim[1] * lc_dim[2] + mc[1] * lc_dim[2] + mc[2]

                    i_particle = head[c_index]
                    while i_particle != EMPTY:
                        pos_i = positions[i_particle]
                        # Iterate through neighbor offsets
                        for offset in neighbor_offsets:
                            mc1 = mc + offset
                            rshift = np.zeros(dim)
                            valid_cell = True

                            # Apply periodic boundary conditions (PBC)
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

                            # Compute the linear index for the neighboring cell
                            c1 = mc1[0] * lc_dim[1] * lc_dim[2] + mc1[1] * lc_dim[2] + mc1[2]

                            j_particle = head[c1]
                            while j_particle != EMPTY:
                                if j_particle > i_particle:
                                    pos_j = positions[j_particle] + rshift
                                    r_ij = pos_i - pos_j
                                    dist = np.linalg.norm(r_ij)

                                    if dist < rcutoff and dist > 1e-12:
                                        f_mag = compute_lj_force(dist, sigma, epsilon, rcutoff)
                                        fij = f_mag * (r_ij / dist)

                                        forces[i_particle] += fij
                                        forces[j_particle] -= fij

                                        potential_energy += compute_lj_potential(dist, sigma, epsilon, rcutoff)
                                j_particle = lscl[j_particle]
                        i_particle = lscl[i_particle]

    return forces, potential_energy