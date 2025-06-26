import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic

from model import *
N_PARTICLES  = 10100 # Denis wants more particles 
N_STEPS      = 100

def leap_frog_step(x, y, z, vx, vy, vz, xp, yp, zp, logM, Rs, q, dirx, diry, dirz, logm, rs, dt):
    """Perform a single leapfrog step for the orbit integration."""
    # Calculate the acceleration    
    ax, ay, az = scalar_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    scalar_Plummer_acceleration(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) # km2 / s / Gyr / kpc

    # Update velocities
    vx += 0.5 * ax * dt * KPC_TO_KM**-1
    vy += 0.5 * ay * dt * KPC_TO_KM**-1
    vz += 0.5 * az * dt * KPC_TO_KM**-1
    
    # Update positions
    x += vx * dt * GYR_TO_S * KPC_TO_KM**-1
    y += vy * dt * GYR_TO_S * KPC_TO_KM**-1
    z += vz * dt * GYR_TO_S * KPC_TO_KM**-1
    
    # Recalculate the acceleration at the new position
    ax, ay, az = scalar_NFW_acceleration(x, y, z, logM, Rs, q, dirx, diry, dirz) +  \
                    scalar_Plummer_acceleration(x, y, z, logm, rs, x_origin=xp, y_origin=yp, z_origin=zp) # km2 / s / Gyr / kpc
    
    # Final update of velocities
    vx += 0.5 * ax * dt * KPC_TO_KM**-1
    vy += 0.5 * ay * dt * KPC_TO_KM**-1
    vz += 0.5 * az * dt * KPC_TO_KM**-1
    
    return x, y, z, vx, vy, vz

def condensed_track(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha=1.0, tail=0, r_thresh=5, N_thresh=5):
    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                            logM, Rs, q, dirx, diry, dirz,
                                            time)
    theta_sat = jnp.arctan2(xv_sat[:, 1], xv_sat[:, 0])
    theta_sat = jnp.where(theta_sat < 0, theta_sat + 2 * jnp.pi, theta_sat)
    theta_sat = jax_unwrap(theta_sat)

    xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                logM, Rs, q, dirx, diry, dirz,
                                                time*alpha)

    hessians = vector_NFW_Hessian(xv_sat_forward[:, 0], xv_sat_forward[:, 1], xv_sat_forward[:, 2],
                                    logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, xv_sat_forward, 10 ** logm)

    ic_particle_spray = create_ic_particle_spray(xv_sat_forward, rj, vj, R, tail)

    time_steps = np.linspace(0, time, N_STEPS)

    xv_stream = np.zeros((N_STEPS, N_PARTICLES, 7)) - np.nan
    weight = np.ones((N_STEPS, N_PARTICLES))

    time_steps = np.linspace(0, time, N_STEPS)

    xv_stream = np.zeros((N_STEPS, N_PARTICLES, 7)) - np.nan
    weight = np.ones((N_STEPS, N_PARTICLES))

    list_n_steps = []
    particules_leading = []
    particules_trailing = []
    unwrap_counter = np.zeros(N_PARTICLES)
    for index in tqdm(range(len(time_steps)-1), leave=True):
        xv_sat_here = xv_sat_forward[index]
        index_range = (N_PARTICLES//N_STEPS)
        xv_stream[index, index_range*index:index_range*(index+1), :-1] = ic_particle_spray[index_range*index:index_range*(index+1)]

        n_steps = 0
        for j in range(N_PARTICLES):
            if ~np.isnan(xv_stream[index, j, 0]):
                n_steps += 1
                xv_stream[index+1, j, :-1] = leap_frog_step(xv_stream[index, j, 0], xv_stream[index, j, 1], xv_stream[index, j, 2], xv_stream[index, j, 3], xv_stream[index, j, 4], xv_stream[index, j, 5],
                                                                                        xv_sat_here[0], xv_sat_here[1], xv_sat_here[2],
                                                                                        logM, Rs, q, dirx, diry, dirz, logm, rs, dt=time_steps[1])
        # print(f'Computed {orbits_computes} orbits for step {index+1}/{len(time_steps)-1}')

                theta_stream = np.arctan2(xv_stream[index+1, j, 1], xv_stream[index+1, j, 0])
                theta_stream = np.where(theta_stream < 0, theta_stream + 2 * np.pi, theta_stream)
                if np.isnan(xv_stream[index, j, -1]) or (xv_stream[index, j, -1] < theta_stream + 2 * np.pi * unwrap_counter[j]):
                    xv_stream[index+1, j, -1] = theta_stream + 2 * np.pi * unwrap_counter[j]
                elif xv_stream[index, j, -1] > theta_stream:
                    unwrap_counter[j] += 1
                    xv_stream[index+1, j, -1] = theta_stream + 2 * np.pi * unwrap_counter[j]
                else:
                    print("Error in theta calculation", xv_stream[index+1, j, -1], theta_stream)


        r_from_sat = np.sqrt(xv_stream[index+1, :, 0]**2 + xv_stream[index+1, :, 1]**2 + xv_stream[index+1, :, 2]**2) - np.sqrt(xv_sat_forward[index+1, 0]**2 + xv_sat_forward[index+1, 1]**2 + xv_sat_forward[index+1, 2]**2)
        r_min_here = r_thresh*rj[index]

        arg_to_bin  = np.where((abs(r_from_sat) > r_min_here) & (weight[index+1] == 1))[0]
        sign_to_bin = np.sign(r_from_sat[arg_to_bin])
        arg_to_bin_leading = arg_to_bin[sign_to_bin < 0]
        arg_to_bin_trailing = arg_to_bin[sign_to_bin > 0]

        if len(arg_to_bin_leading) > N_thresh:
            # print(f'Binning {len(arg_to_bin_leading)} leading particles')
            xv_stream[index+1, arg_to_bin_leading[0]]  = np.mean(xv_stream[index+1, arg_to_bin_leading], axis=0)
            particules_leading.append(xv_stream[index+1, arg_to_bin_leading])
            weight[index+1:, arg_to_bin_leading[0]] = len(arg_to_bin_leading)
            xv_stream[index+1:, arg_to_bin_leading[1:]] = np.nan
            weight[index+1:, arg_to_bin_leading[1:]] = np.nan
        else:
            particules_leading.append([])

        if len(arg_to_bin_trailing) > N_thresh:
            # print(f'Binning {len(arg_to_bin_trailing)} trailing particles')
            xv_stream[index+1, arg_to_bin_trailing[0]] = np.mean(xv_stream[index+1, arg_to_bin_trailing], axis=0)
            particules_trailing.append(xv_stream[index+1, arg_to_bin_trailing])
            weight[index+1:, arg_to_bin_trailing[0]] = len(arg_to_bin_trailing)
            xv_stream[index+1:, arg_to_bin_trailing[1:]] = np.nan
            weight[index+1:, arg_to_bin_trailing[1:]] = np.nan
        else:
            particules_trailing.append([])

        list_n_steps.append(n_steps)
    list_n_steps = [0] + list_n_steps

    # Unwrap
    theta_stream = xv_stream[-1, :, -1] #np.unwrap(np.nan_to_num(xv_stream[:, :, -1], nan=0.0), axis=0)[-1]

    # === Process angles as a function of Progenitor ===
    # Count how many complete 2pi rotations have been accumulated (integer division).
    theta_count = np.floor_divide(theta_sat, 2 * np.pi)

    final_theta_stream = (
        theta_stream #np.sum(theta_stream * diagonal_matrix, axis=1)
        - theta_sat[-1]
        + np.repeat(theta_count,  N_PARTICLES// N_STEPS) * 2 * np.pi
        )

    algin_reference = theta_sat[-1] - theta_count[-1] * (2 * np.pi) # Make sure the angle of reference is at theta=0

    final_theta_stream += (1 - np.sign(algin_reference - np.pi))/2 * algin_reference + \
                            (1 + np.sign(algin_reference - np.pi))/2 * (algin_reference - 2 * np.pi)
    r_stream = np.sqrt(xv_stream[-1, :, 0]**2 + xv_stream[-1, :, 1]**2)
    x_stream = xv_stream[-1, :, 0]
    y_stream = xv_stream[-1, :, 1]


    bin_edges = np.linspace(-2 * np.pi, 2 * np.pi, N_BINS + 1)
    theta_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate histograms and bin counts for the angles (with weights)
    hist, _ = np.histogram(final_theta_stream, bins=bin_edges, weights=weight[-1, :])

    # Compute weighted averages for r_stream, x_stream, y_stream using the same bins
    r_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream * weight[-1, :])[0] / hist
    x_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=x_stream * weight[-1, :])[0] / hist
    y_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=y_stream * weight[-1, :])[0] / hist

    dict_stream = {'theta_stream': final_theta_stream, 'r_stream': r_stream, 'x_stream': x_stream, 'y_stream': y_stream, 'weight': weight[-1, :],
                    'theta_bin': theta_bin, 'r_bin': r_bin, 'x_bin': x_bin, 'y_bin': y_bin,
                    'r_thresh': r_thresh, 'N_thresh': N_thresh, 'n_steps':list_n_steps}
    
    return dict_stream
