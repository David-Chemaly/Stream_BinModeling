import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic

import jax
from model import *
N_PARTICLES  = 10100 # Denis wants more particles 
N_STEPS      = 100

def test_leap_frog_step(theta0, x, y, z, vx, vy, vz, xp, yp, zp, logM, Rs, q, dirx, diry, dirz, logm, rs, dt):
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
    
    theta = jnp.arctan2(y, x)
    theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)

    theta = unwrap_step(theta, theta0)

    return theta, x, y, z, vx, vy, vz

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

def condensed_track(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail=0, r_thresh=5, N_thresh=5, percentiles=[16, 84], seed=111, vectorized=False):
    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                            logM, Rs, q, dirx, diry, dirz,
                                            time)

    xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                logM, Rs, q, dirx, diry, dirz,
                                                time*alpha)
    theta_sat_forward = jnp.arctan2(xv_sat_forward[:, 1], xv_sat_forward[:, 0])
    theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + 2 * jnp.pi, theta_sat_forward)
    theta_sat_forward = jax_unwrap(theta_sat_forward)
    
    hessians = vector_NFW_Hessian(xv_sat_forward[:, 0], xv_sat_forward[:, 1], xv_sat_forward[:, 2],
                                    logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, xv_sat_forward, 10 ** logm)

    ic_particle_spray = create_ic_particle_spray(xv_sat_forward, rj, vj, R, tail, seed=seed)

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

        if vectorized:
            n_steps = np.sum(~np.isnan(xv_stream[index, :, 0]))
            xv_stream[index+1, :, :-1] = np.stack(jax.vmap(leap_frog_step, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None)) \
                                            (xv_stream[index, :, 0], xv_stream[index, :, 1], xv_stream[index, :, 2], xv_stream[index, :, 3], xv_stream[index, :, 4], xv_stream[index, :, 5],
                                            xv_sat_here[0], xv_sat_here[1], xv_sat_here[2],
                                            logM, Rs, q, dirx, diry, dirz, logm, rs, time_steps[1])).T

            theta = np.arctan2(xv_stream[index+1, :, 1], xv_stream[index+1, :, 0])
            theta = np.where(theta < 0, theta + 2*np.pi, theta)

            prev = xv_stream[index, :, -1]
            is_nan = np.isnan(prev)

            mask_ok  = is_nan | (prev < theta + 2*np.pi * unwrap_counter)
            mask_add = (~mask_ok) & (prev > theta)

            xv_stream[index+1, mask_ok, -1] = theta[mask_ok] + 2*np.pi * unwrap_counter[mask_ok]

            unwrap_counter[mask_add] += 1
            xv_stream[index+1, mask_add, -1] = theta[mask_add] + 2*np.pi * unwrap_counter[mask_add]


            # theta_stream = np.arctan2(xv_stream[index+1, :, 1], xv_stream[index+1, :, 0])
            # theta_stream = np.where(theta_stream < 0, theta_stream + 2 * np.pi, theta_stream)

            # mask_ok  = np.isnan(xv_stream[index, :, -1]) | (xv_stream[index, :, -1] < theta_stream + 2 * np.pi * unwrap_counter[:])
            # mask_add = (xv_stream[index, :, -1] > theta_stream) & ~mask_ok 
            # if mask_ok.any():
            #     xv_stream[index+1, mask_ok, -1] = theta_stream[mask_ok] + 2 * np.pi * unwrap_counter[mask_ok]
            # if mask_add.any():
            #     unwrap_counter[mask_add] += 1
            #     xv_stream[index+1, mask_add, -1] = theta_stream[mask_add] + 2 * np.pi * unwrap_counter[mask_add] + 2 * np.pi

        else:
            n_steps = 0
            for j in range(N_PARTICLES):
                if ~np.isnan(xv_stream[index, j, 0]):
                    n_steps += 1
                    xv_stream[index+1, j, :-1] = leap_frog_step(xv_stream[index, j, 0], xv_stream[index, j, 1], xv_stream[index, j, 2], xv_stream[index, j, 3], xv_stream[index, j, 4], xv_stream[index, j, 5],
                                                                                            xv_sat_here[0], xv_sat_here[1], xv_sat_here[2],
                                                                                            logM, Rs, q, dirx, diry, dirz, logm, rs, dt=time_steps[1])

                    theta_stream = np.arctan2(xv_stream[index+1, j, 1], xv_stream[index+1, j, 0])
                    theta_stream = np.where(theta_stream < 0, theta_stream + 2 * np.pi, theta_stream)
                    if np.isnan(xv_stream[index, j, -1]) or (xv_stream[index, j, -1] < theta_stream + 2 * np.pi * unwrap_counter[j]):
                        xv_stream[index+1, j, -1] = theta_stream + 2 * np.pi * unwrap_counter[j]
                    elif xv_stream[index, j, -1] > theta_stream:
                        unwrap_counter[j] += 1
                        xv_stream[index+1, j, -1] = theta_stream + 2 * np.pi * unwrap_counter[j]
                    else:
                        print("Error in theta calculation", xv_stream[index+1, j, -1], theta_stream)


        r_from_sat = np.sqrt((xv_stream[index+1, :, 0] - xv_sat_forward[index+1, 0])**2 
                            +(xv_stream[index+1, :, 1] - xv_sat_forward[index+1, 1])**2 
                            +(xv_stream[index+1, :, 2] - xv_sat_forward[index+1, 2])**2 
        )
        r_min_here = r_thresh*rj[index]
        arg_to_bin  = np.where((abs(r_from_sat) > r_min_here) & (abs(r_from_sat) > r_min_here) & (weight[index+1] == 1))[0]

        #Gibbons et al. 20XX
        r_from_sat = np.sqrt(xv_stream[index+1, :, 0]**2 + xv_stream[index+1, :, 1]**2 + xv_stream[index+1, :, 2]**2) \
                                - np.sqrt(xv_sat_forward[index+1, 0]**2 + xv_sat_forward[index+1, 1]**2 + xv_sat_forward[index+1, 2]**2)
        
        sign_to_bin = np.sign(r_from_sat[arg_to_bin])
        arg_to_bin_leading = arg_to_bin[sign_to_bin < 0]
        arg_to_bin_trailing = arg_to_bin[sign_to_bin > 0]

        if len(arg_to_bin_leading) > N_thresh:
            if len(percentiles) != 0:
                vt_leading = np.sqrt(xv_stream[index+1, arg_to_bin_leading, 3]**2 + xv_stream[index+1, arg_to_bin_leading, 4]**2 
                                      - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2))
                # vt_leading = np.sqrt((xv_stream[index+1, arg_to_bin_leading, 3] - xv_sat_forward[index+1, 3])**2 
                #                     +(xv_stream[index+1, arg_to_bin_leading, 4] - xv_sat_forward[index+1, 4])**2)
                # vt_leading = np.dot(xv_stream[index+1, arg_to_bin_leading, 3:6], xv_sat_forward[index+1, 3:6]) / np.linalg.norm(xv_sat_forward[index+1, 3:6]) # Projected velocity in the direction of the satellite
                values = np.percentile(vt_leading, percentiles)

                # Loop over percentiles to handle all ranges
                for p in range(len(percentiles)+1):

                    if p == 0:
                        arg_here = np.where(vt_leading <= values[p])[0]
                    elif p == len(percentiles):
                        arg_here = np.where(values[p-1] < vt_leading)[0]
                    else:
                        arg_here = np.where( ( values[p-1] < vt_leading) & (vt_leading <= values[p]) )[0]

                    xv_stream[index+1, arg_to_bin_leading[p]] = np.median(xv_stream[index+1, arg_to_bin_leading[arg_here]], axis=0)
                    weight[index+1:, arg_to_bin_leading[p]] = len(arg_here)

                # Update the bins that are not part of the specified percentiles
                xv_stream[index+1:, arg_to_bin_leading[len(percentiles)+1:]] = np.nan
                weight[index+1:, arg_to_bin_leading[len(percentiles)+1:]] = np.nan

                particules_leading.append(xv_stream[index+1, arg_to_bin_leading])
            else:
                # print(f'Binning {len(arg_to_bin_leading)} leading particles')
                xv_stream[index+1, arg_to_bin_leading[0]]  = np.median(xv_stream[index+1, arg_to_bin_leading], axis=0)
                particules_leading.append(xv_stream[index+1, arg_to_bin_leading])
                weight[index+1:, arg_to_bin_leading[0]] = len(arg_to_bin_leading)
                xv_stream[index+1:, arg_to_bin_leading[1:]] = np.nan
                weight[index+1:, arg_to_bin_leading[1:]] = np.nan
        else:
            particules_leading.append([])

        if len(arg_to_bin_trailing) > N_thresh:
            if len(percentiles) != 0:
                vt_trailing = np.sqrt(xv_stream[index+1, arg_to_bin_trailing, 3]**2 + xv_stream[index+1, arg_to_bin_trailing, 4]**2 
                                      - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2))
                # vt_trailing = np.sqrt((xv_stream[index+1, arg_to_bin_trailing, 3] - xv_sat_forward[index+1, 3])**2 
                #                      +(xv_stream[index+1, arg_to_bin_trailing, 4] - xv_sat_forward[index+1, 4])**2)
                # vt_trailing = np.dot(xv_stream[index+1, arg_to_bin_trailing, 3:6], xv_sat_forward[index+1, 3:6]) / np.linalg.norm(xv_sat_forward[index+1, 3:6])
                values = np.percentile(vt_trailing, percentiles)

                # Loop over percentiles to handle all ranges
                for p in range(len(percentiles)+1):

                    if p == 0:
                        arg_here = np.where(vt_trailing <= values[p])[0]
                    elif p == len(percentiles):
                        arg_here = np.where(values[p-1] < vt_trailing)[0]
                    else:
                        arg_here = np.where( ( values[p-1] < vt_trailing) & (vt_trailing <= values[p]) )[0]

                    xv_stream[index+1, arg_to_bin_trailing[p]] = np.median(xv_stream[index+1, arg_to_bin_trailing[arg_here]], axis=0)
                    weight[index+1:, arg_to_bin_trailing[p]] = len(arg_here)

                # Update the bins that are not part of the specified percentiles
                xv_stream[index+1:, arg_to_bin_trailing[len(percentiles)+1:]] = np.nan
                weight[index+1:, arg_to_bin_trailing[len(percentiles)+1:]] = np.nan

                particules_trailing.append(xv_stream[index+1, arg_to_bin_trailing])
            else:
                # print(f'Binning {len(arg_to_bin_trailing)} trailing particles')
                xv_stream[index+1, arg_to_bin_trailing[0]] = np.median(xv_stream[index+1, arg_to_bin_trailing], axis=0)
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
    theta_count = np.floor_divide(theta_sat_forward, 2 * np.pi)

    final_theta_stream = (
        theta_stream #np.sum(theta_stream * diagonal_matrix, axis=1)
        - theta_sat_forward[-1]
        + np.repeat(theta_count,  N_PARTICLES// N_STEPS) * 2 * np.pi
        )

    algin_reference = theta_sat_forward[-1] - theta_count[-1] * (2 * np.pi) # Make sure the angle of reference is at theta=0

    final_theta_stream += (1 - np.sign(algin_reference - np.pi))/2 * algin_reference + \
                            (1 + np.sign(algin_reference - np.pi))/2 * (algin_reference - 2 * np.pi)
    r_stream = np.sqrt(xv_stream[-1, :, 0]**2 + xv_stream[-1, :, 1]**2)
    x_stream = xv_stream[-1, :, 0]
    y_stream = xv_stream[-1, :, 1]


    bin_edges = np.linspace(-2 * np.pi, 2 * np.pi, N_BINS + 1)
    theta_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate histograms and bin counts for the angles (with weights)
    hist, _ = np.histogram(final_theta_stream, bins=bin_edges, weights=weight[-1, :])
    hist[hist == 0] = np.nan

    # Compute weighted averages for r_stream, x_stream, y_stream using the same bins
    r_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream * weight[-1, :])[0] / hist
    x_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=x_stream * weight[-1, :])[0] / hist
    y_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=y_stream * weight[-1, :])[0] / hist

    w_bin = np.sqrt(np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream**2 * weight[-1, :])[0] / hist - r_bin**2)

    dict_stream = {'theta_stream': final_theta_stream, 'r_stream': r_stream, 'x_stream': x_stream, 'y_stream': y_stream, 'weight_stream': weight[-1, :],
                    'theta_bin': theta_bin, 'r_bin': r_bin, 'w_bin': w_bin, 'x_bin': x_bin, 'y_bin': y_bin, 'weight_bin': hist, 'r_sig_bin': w_bin/np.sqrt(hist),
                    'r_thresh': r_thresh, 'N_thresh': N_thresh, 'n_steps':list_n_steps, 'xv_stream': xv_stream, 'xv_sat': xv_sat_forward}
    
    return dict_stream

def vectorized_condensed_track(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail=0, min_count=101, r_thresh=5, percentiles=[16, 84], seed=111, verbose=True):
    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                            logM, Rs, q, dirx, diry, dirz,
                                            time)

    xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                logM, Rs, q, dirx, diry, dirz,
                                                time*alpha)
    theta_sat_forward = jnp.arctan2(xv_sat_forward[:, 1], xv_sat_forward[:, 0])
    theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + 2 * jnp.pi, theta_sat_forward)
    theta_sat_forward = jax_unwrap(theta_sat_forward)
    
    hessians = vector_NFW_Hessian(xv_sat_forward[:, 0], xv_sat_forward[:, 1], xv_sat_forward[:, 2],
                                    logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, xv_sat_forward, 10 ** logm)

    ic_particle_spray = create_ic_particle_spray(xv_sat_forward, rj, vj, R, tail, seed=seed)

    time_steps = np.linspace(0, time, N_STEPS)

    list_n_steps = []

    xv_stream = np.zeros((N_STEPS, N_PARTICLES, 7)) - np.nan
    weight = np.ones((N_STEPS, N_PARTICLES)) #- np.nan

    index_range = (N_PARTICLES//N_STEPS)
    for index in tqdm(range(len(time_steps)-1), leave=True, disable=verbose):

        xv_stream[index, index_range*index:index_range*(index+1), 1:] = ic_particle_spray[index_range*index:index_range*(index+1)]

        theta0 = np.arctan2(ic_particle_spray[index_range*index:index_range*(index+1), 1], ic_particle_spray[index_range*index:index_range*(index+1), 0])
        theta0 = np.where(theta0 < 0, theta0 + 2 * np.pi, theta0)
        xv_stream[index, index_range*index:index_range*(index+1), 0] = theta0

        xv_sat_here  = xv_sat_forward[index]
        xv_sat_futur = xv_sat_forward[index+1]

        xv_stream_here = xv_stream[index, :index_range*(index+1)]

        live    = ~np.isnan(xv_stream_here[:, 0])
        n_steps = np.sum(live)
        xv_stream_here_and_live = xv_stream_here#[live]
        xv_stream_futur = np.stack(jax.vmap(test_leap_frog_step, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None)) \
                            (xv_stream_here_and_live[:, 0], xv_stream_here_and_live[:, 1], xv_stream_here_and_live[:, 2],xv_stream_here_and_live[:, 3], xv_stream_here_and_live[:, 4], xv_stream_here_and_live[:, 5], xv_stream_here_and_live[:, 6],
                            xv_sat_here[0], xv_sat_here[1], xv_sat_here[2], logM, Rs, q, dirx, diry, dirz, logm, rs, time_steps[1])).T
        xv_stream_futur_and_live = xv_stream_futur[live]
        xv_stream[index+1, :index_range*(index+1)][live] = xv_stream_futur_and_live

        # Checking who escaped for binning
        r_from_sat = np.sqrt((xv_stream_futur_and_live[:, 1] - xv_sat_futur[0])**2 
                            +(xv_stream_futur_and_live[:, 2] - xv_sat_futur[1])**2 
                            +(xv_stream_futur_and_live[:, 3] - xv_sat_futur[2])**2 
        )
        r_min_here  = r_thresh*rj[index]
        # Vectorized mask for escaped particles
        mask = (np.abs(r_from_sat) > r_min_here) & (~np.isnan(r_from_sat)) & (weight[index+1, :index_range*(index+1)][live] == 1)
        arg_to_bin = np.flatnonzero(mask)

        #Gibbons et al. 20XX
        r_from_sat = np.sqrt(xv_stream_futur_and_live[:, 0]**2 + xv_stream_futur_and_live[:, 1]**2 + xv_stream_futur_and_live[:, 2]**2) \
                                - np.sqrt(xv_sat_forward[index+1, 0]**2 + xv_sat_forward[index+1, 1]**2 + xv_sat_forward[index+1, 2]**2)

        sign_to_bin = np.sign(r_from_sat[arg_to_bin])
        arg_to_bin_leading = arg_to_bin[sign_to_bin < 0]
        arg_to_bin_trailing = arg_to_bin[sign_to_bin > 0]

        # Bin leading 
        if len(arg_to_bin_leading) > len(percentiles):
            vt_leading = np.sqrt(xv_stream_futur_and_live[arg_to_bin_leading, 4]**2 + xv_stream_futur_and_live[arg_to_bin_leading, 5]**2) \
                            - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2)
            
            cuts = np.percentile(vt_leading, percentiles)            # shape (len(percentiles),)
            bins = np.concatenate(([-np.inf], cuts, [np.inf]))       # shape (len(percentiles)+2,)
            vt_bins = np.digitize(vt_leading, bins) - 1               # bins labeled 0..n_bins-1

            n_bins = len(percentiles) + 1                                  # number of bins
            counts  = np.bincount(vt_bins, minlength=n_bins)  # → (n_bins,)

            stop = index_range * (index + 1)
            live_cols = np.flatnonzero(live[:stop])
            cols = live_cols[arg_to_bin_leading]
            for b in range(n_bins):
                xv_stream[index+1, cols[b]] = np.median(xv_stream_futur_and_live[arg_to_bin_leading, :][vt_bins == b], axis=0)
                weight[index+1:, cols[b]]   = counts[b]

            xv_stream[index+1:, cols[n_bins:]] = np.nan
            weight[index+1:, cols[n_bins:]] = np.nan

        # Bin trailing
        if len(arg_to_bin_trailing) > len(percentiles):
            vt_trailing = np.sqrt(xv_stream_futur_and_live[arg_to_bin_trailing, 4]**2 + xv_stream_futur_and_live[arg_to_bin_trailing, 5]**2) \
                            - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2)
            
            cuts = np.percentile(vt_trailing, percentiles)            # shape (len(percentiles),)
            bins = np.concatenate(([-np.inf], cuts, [np.inf]))       # shape (len(percentiles)+2,)
            vt_bins = np.digitize(vt_trailing, bins) - 1               # bins labeled 0..n_bins-1

            n_bins = len(percentiles) + 1                                  # number of bins
            counts  = np.bincount(vt_bins, minlength=n_bins)  # → (n_bins,)

            stop = index_range * (index + 1)
            live_cols = np.flatnonzero(live[:stop])
            cols = live_cols[arg_to_bin_trailing]
            for b in range(n_bins):
                xv_stream[index+1, cols[b]] = np.median(xv_stream_futur_and_live[arg_to_bin_trailing, :][vt_bins == b], axis=0)
                weight[index+1:, cols[b]]   = counts[b]

            xv_stream[index+1:, cols[n_bins:]] = np.nan
            weight[index+1:, cols[n_bins:]] = np.nan

        list_n_steps.append(n_steps)
    list_n_steps = [0] + list_n_steps

    # Unwrap
    theta_stream = xv_stream[-1, :, 0] #np.unwrap(np.nan_to_num(xv_stream[:, :, -1], nan=0.0), axis=0)[-1]

    # === Process angles as a function of Progenitor ===
    # Count how many complete 2pi rotations have been accumulated (integer division).
    theta_count = np.floor_divide(theta_sat_forward, 2 * np.pi)

    final_theta_stream = (
        theta_stream #np.sum(theta_stream * diagonal_matrix, axis=1)
        - theta_sat_forward[-1]
        + np.repeat(theta_count,  N_PARTICLES// N_STEPS) * 2 * np.pi
        )

    algin_reference = theta_sat_forward[-1] - theta_count[-1] * (2 * np.pi) # Make sure the angle of reference is at theta=0

    final_theta_stream += (1 - np.sign(algin_reference - np.pi))/2 * algin_reference + \
                            (1 + np.sign(algin_reference - np.pi))/2 * (algin_reference - 2 * np.pi)
    r_stream = np.sqrt(xv_stream[-1, :, 1]**2 + xv_stream[-1, :, 2]**2)
    x_stream = xv_stream[-1, :, 1]
    y_stream = xv_stream[-1, :, 2]

    bin_edges = np.linspace(-2 * np.pi, 2 * np.pi, N_BINS + 1)
    theta_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate histograms and bin counts for the angles (with weights)
    hist, _ = np.histogram(final_theta_stream, bins=bin_edges, weights=weight[-1, :])
    hist[hist == 0] = np.nan

    # Compute weighted averages for r_stream, x_stream, y_stream using the same bins
    r_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream * weight[-1, :])[0] / hist
    x_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=x_stream * weight[-1, :])[0] / hist
    y_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=y_stream * weight[-1, :])[0] / hist

    # Set really small values (e.g., < 1e-10) to zero for numerical stability
    w_bin_raw = np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream**2 * weight[-1, :])[0] / hist - r_bin**2
    w_bin_raw[ (-1e-5 < w_bin_raw) * (w_bin_raw < 0)] = 0.0
    w_bin = np.sqrt(w_bin_raw)
    r_sig_bin = w_bin/np.sqrt(hist)

    # Mask bins with insufficient counts only once
    mask = hist > min_count
    theta_bin[~mask] = np.nan
    r_bin[~mask] = np.nan
    w_bin[~mask] = np.nan
    x_bin[~mask] = np.nan
    y_bin[~mask] = np.nan
    r_sig_bin[~mask] = np.nan

    dict_stream = {'theta_stream': final_theta_stream, 'r_stream': r_stream, 'x_stream': x_stream, 'y_stream': y_stream, 'weight_stream': weight[-1, :],
                    'theta_bin': theta_bin, 'r_bin': r_bin, 'w_bin': w_bin, 'x_bin': x_bin, 'y_bin': y_bin, 'weight_bin': hist, 'r_sig_bin': r_sig_bin,
                    'r_thresh': r_thresh, 'n_steps':list_n_steps, 'xv_stream': xv_stream, 'xv_sat': xv_sat_forward}
    
    return dict_stream

def jax_condensed_track(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail=0, r_thresh=5, N_thresh=5, percentiles=[16, 84], seed=111):
    
    @jax.jit
    def forward_integrate_stream_leapfrog(index, x0, y0, z0, vx0, vy0, vz0,
                                            xv_sat, logM, Rs, q,
                                            dirx, diry, dirz, logm, rs, time):
        # State is a flat tuple of six scalars.
        xp, yp, zp, vxp, vyp, vzp = xv_sat[index]

        theta0 = jnp.arctan2(y0, x0)
        theta0 = jax.lax.cond(theta0 < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta0)

        state = (theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
        dt_sat = time / N_STEPS

        time_here = time - index * dt_sat
        dt_here = time_here / N_STEPS

        def step_fn(state, _):
            # Use only the first three elements of the satellite row.
            theta0, x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp = state

            initial_conditions = (x0, y0, z0, vx0, vy0, vz0, xp, yp, zp, vxp, vyp, vzp)
            final_conditions = leapfrog_stream_step(initial_conditions, dt_here,
                                                logM, Rs, q, dirx, diry, dirz, logm, rs)
            
            theta = jnp.arctan2(final_conditions[1], final_conditions[0])
            theta = jax.lax.cond(theta < 0, lambda x: x + 2 * jnp.pi, lambda x: x, theta)

            theta = unwrap_step(theta, theta0)

            new_state = (theta, *final_conditions)

            # The carry and output must have the same structure.
            return new_state, _ # jnp.stack(new_state)

        # Run integration over the satellite trajectory (using all but the last row).
        trajectory, _ = jax.lax.scan(step_fn, state, None, length=1, unroll=True)
        # 'trajectory' is a tuple of six arrays, each of shape (N_STEPS,).

        return jnp.array(trajectory)

    @jax.jit
    def generate_stream(ic_particle_spray, xv_sat, logM, Rs, q,
                        dirx, diry, dirz, logm, rs, time):
        # There are 16 parameters to forward_integrate_stream_leapfrog:
        # 6 come from ic_particle_spray (one per coordinate),
        # and the remaining 10 are shared (xv_sat, logM, Rs, q, dirx, diry, dirz, logm, rs, time).
        index = jnp.repeat(jnp.arange(0, N_STEPS, 1), N_PARTICLES// N_STEPS)  # Shape: (N_PARTICLES,)

        xv_stream = jax.vmap(
            forward_integrate_stream_leapfrog,
            in_axes=(0, 0, 0, 0, 0, 0, 0,  # map over each column of ic_particle_spray
                        None, None, None, None, None, None, None, None, None, None)  # shared arguments
        )(index,
            ic_particle_spray[:, 0],  # x0
            ic_particle_spray[:, 1],  # y0
            ic_particle_spray[:, 2],  # z0
            ic_particle_spray[:, 3],  # vx0
            ic_particle_spray[:, 4],  # vy0
            ic_particle_spray[:, 5],  # vz0
            xv_sat, # (xp, yp, zp, vxp, vyp, vzp)
            logM, Rs, q,
            dirx, diry, dirz, logm, rs, time)

        return xv_stream


    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0,
                                        logM, Rs, q, dirx, diry, dirz,
                                        time)

    xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                logM, Rs, q, dirx, diry, dirz,
                                                time*alpha)
    theta_sat_forward = jnp.arctan2(xv_sat_forward[:, 1], xv_sat_forward[:, 0])
    theta_sat_forward = jnp.where(theta_sat_forward < 0, theta_sat_forward + 2 * jnp.pi, theta_sat_forward)
    theta_sat_forward = jax_unwrap(theta_sat_forward)

    hessians = vector_NFW_Hessian(xv_sat_forward[:, 0], xv_sat_forward[:, 1], xv_sat_forward[:, 2],
                                    logM, Rs, q, dirx, diry, dirz)
    rj, vj, R = get_rj_vj_R(hessians, xv_sat_forward, 10 ** logm)

    ic_particle_spray = create_ic_particle_spray(xv_sat_forward, rj, vj, R, tail, seed=seed)

    time_steps = np.linspace(0, time, N_STEPS)

    xv_stream = np.zeros((N_STEPS, N_PARTICLES, 7)) - np.nan
    xv_stream[:, :, 0] = np.arctan2(ic_particle_spray[:, 1], ic_particle_spray[:, 0])
    xv_stream[:, :, 0] = np.where(xv_stream[:, :, 0] < 0, xv_stream[:, :, 0] + 2 * np.pi, xv_stream[:, :, 0])
    xv_stream[0, :, 1:] = ic_particle_spray

    weight = np.ones((N_STEPS, N_PARTICLES))

    list_n_steps = []
    particules_leading = []
    particules_trailing = []
    unwrap_counter = np.zeros(N_PARTICLES)

    start_xv = ic_particle_spray.copy()

    for index in tqdm(range(len(time_steps)-1), leave=True):
        end_xv = generate_stream(xv_stream[index, :, 1:], xv_sat_forward, logM, Rs, q, dirx, diry, dirz, logm, rs, time)

        index_range = (N_PARTICLES // N_STEPS)
        xv_stream[index+1, :, :] = end_xv[:, :7]

        xv_stream_so_far = xv_stream[index+1, :index_range*(index+2)]
        weight_so_far = weight[index+1, :index_range*(index+2)]

        r_from_sat = np.sqrt((xv_stream_so_far[:, 1] - xv_sat_forward[index+1, 0])**2 
                            +(xv_stream_so_far[:, 2] - xv_sat_forward[index+1, 1])**2 
                            +(xv_stream_so_far[:, 3] - xv_sat_forward[index+1, 2])**2 
        )
        r_min_here = r_thresh*rj[index]
        arg_to_bin  = np.where((abs(r_from_sat) > r_min_here) & (abs(r_from_sat) > r_min_here) & (weight_so_far == 1))[0]

        #Gibbons et al. 20XX
        r_from_sat = np.sqrt(xv_stream_so_far[:, 1]**2 + xv_stream_so_far[:, 2]**2 + xv_stream_so_far[:, 3]**2) \
                                - np.sqrt(xv_sat_forward[index+1, 0]**2 + xv_sat_forward[index+1, 1]**2 + xv_sat_forward[index+1, 2]**2)
        
        sign_to_bin = np.sign(r_from_sat[arg_to_bin])
        arg_to_bin_leading = arg_to_bin[sign_to_bin < 0]
        arg_to_bin_trailing = arg_to_bin[sign_to_bin > 0]

        if len(arg_to_bin_leading) > N_thresh:
            if len(percentiles) != 0:
                vt_leading = np.sqrt(xv_stream_so_far[arg_to_bin_leading, 4]**2 + xv_stream_so_far[arg_to_bin_leading, 5]**2 
                                        - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2))
                values = np.percentile(vt_leading, percentiles)

                # Loop over percentiles to handle all ranges
                for p in range(len(percentiles)+1):

                    if p == 0:
                        arg_here = np.where(vt_leading <= values[p])[0]
                    elif p == len(percentiles):
                        arg_here = np.where(values[p-1] < vt_leading)[0]
                    else:
                        arg_here = np.where( ( values[p-1] < vt_leading) & (vt_leading <= values[p]) )[0]

                    xv_stream[index+1, arg_to_bin_leading[p]] = np.median(xv_stream[index+1, arg_to_bin_leading[arg_here]], axis=0)
                    weight[index+1:, arg_to_bin_leading[p]] = len(arg_here)

                # Update the bins that are not part of the specified percentiles
                xv_stream[index+1:, arg_to_bin_leading[len(percentiles)+1:]] = np.nan
                weight[index+1:, arg_to_bin_leading[len(percentiles)+1:]] = np.nan

                particules_leading.append(xv_stream[index+1, arg_to_bin_leading])
            else:
                # print(f'Binning {len(arg_to_bin_leading)} leading particles')
                xv_stream[index+1, arg_to_bin_leading[0]]  = np.median(xv_stream[index+1, arg_to_bin_leading], axis=0)
                particules_leading.append(xv_stream[index+1, arg_to_bin_leading])
                weight[index+1:, arg_to_bin_leading[0]] = len(arg_to_bin_leading)
                xv_stream[index+1:, arg_to_bin_leading[1:]] = np.nan
                weight[index+1:, arg_to_bin_leading[1:]] = np.nan
        else:
            particules_leading.append([])

        if len(arg_to_bin_trailing) > N_thresh:
            if len(percentiles) != 0:
                vt_trailing = np.sqrt(xv_stream_so_far[arg_to_bin_trailing, 4]**2 + xv_stream_so_far[arg_to_bin_trailing, 5]**2 
                                        - np.sqrt(xv_sat_forward[index+1, 3]**2 + xv_sat_forward[index+1, 4]**2))
                # vt_trailing = np.sqrt((xv_stream[index+1, arg_to_bin_trailing, 3] - xv_sat_forward[index+1, 3])**2 
                #                      +(xv_stream[index+1, arg_to_bin_trailing, 4] - xv_sat_forward[index+1, 4])**2)
                # vt_trailing = np.dot(xv_stream[index+1, arg_to_bin_trailing, 3:6], xv_sat_forward[index+1, 3:6]) / np.linalg.norm(xv_sat_forward[index+1, 3:6])
                values = np.percentile(vt_trailing, percentiles)

                # Loop over percentiles to handle all ranges
                for p in range(len(percentiles)+1):

                    if p == 0:
                        arg_here = np.where(vt_trailing <= values[p])[0]
                    elif p == len(percentiles):
                        arg_here = np.where(values[p-1] < vt_trailing)[0]
                    else:
                        arg_here = np.where( ( values[p-1] < vt_trailing) & (vt_trailing <= values[p]) )[0]

                    xv_stream[index+1, arg_to_bin_trailing[p]] = np.median(xv_stream[index+1, arg_to_bin_trailing[arg_here]], axis=0)
                    weight[index+1:, arg_to_bin_trailing[p]] = len(arg_here)

                # Update the bins that are not part of the specified percentiles
                xv_stream[index+1:, arg_to_bin_trailing[len(percentiles)+1:]] = np.nan
                weight[index+1:, arg_to_bin_trailing[len(percentiles)+1:]] = np.nan

                particules_trailing.append(xv_stream[index+1, arg_to_bin_trailing])
            else:
                # print(f'Binning {len(arg_to_bin_trailing)} trailing particles')
                xv_stream[index+1, arg_to_bin_trailing[0]] = np.median(xv_stream[index+1, arg_to_bin_trailing], axis=0)
                particules_trailing.append(xv_stream[index+1, arg_to_bin_trailing])
                weight[index+1:, arg_to_bin_trailing[0]] = len(arg_to_bin_trailing)
                xv_stream[index+1:, arg_to_bin_trailing[1:]] = np.nan
                weight[index+1:, arg_to_bin_trailing[1:]] = np.nan
        else:
            particules_trailing.append([])

    # Unwrap
    theta_stream = np.unwrap(xv_stream[:, :, 0], axis=0)[-1] #np.unwrap(np.nan_to_num(xv_stream[:, :, -1], nan=0.0), axis=0)[-1]

    # === Process angles as a function of Progenitor ===
    # Count how many complete 2pi rotations have been accumulated (integer division).
    theta_count = np.floor_divide(theta_sat_forward, 2 * np.pi)

    final_theta_stream = (
        theta_stream #np.sum(theta_stream * diagonal_matrix, axis=1)
        - theta_sat_forward[-1]
        + np.repeat(theta_count,  N_PARTICLES// N_STEPS) * 2 * np.pi
        )

    algin_reference = theta_sat_forward[-1] - theta_count[-1] * (2 * np.pi) # Make sure the angle of reference is at theta=0

    final_theta_stream += (1 - np.sign(algin_reference - np.pi))/2 * algin_reference + \
                            (1 + np.sign(algin_reference - np.pi))/2 * (algin_reference - 2 * np.pi)
    r_stream = np.sqrt(xv_stream[-1, :, 1]**2 + xv_stream[-1, :, 2]**2)
    x_stream = xv_stream[-1, :, 1]
    y_stream = xv_stream[-1, :, 2]


    bin_edges = np.linspace(-2 * np.pi, 2 * np.pi, N_BINS + 1)
    theta_bin = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate histograms and bin counts for the angles (with weights)
    hist, _ = np.histogram(final_theta_stream, bins=bin_edges, weights=weight[-1, :])

    # Compute weighted averages for r_stream, x_stream, y_stream using the same bins
    r_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream * weight[-1, :])[0] / hist
    x_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=x_stream * weight[-1, :])[0] / hist
    y_bin = np.histogram(final_theta_stream, bins=bin_edges, weights=y_stream * weight[-1, :])[0] / hist

    w_bin = np.sqrt(np.histogram(final_theta_stream, bins=bin_edges, weights=r_stream**2 * weight[-1, :])[0] / hist - r_bin**2)

    dict_stream = {'theta_stream': final_theta_stream, 'r_stream': r_stream, 'x_stream': x_stream, 'y_stream': y_stream, 'weight_stream': weight[-1, :],
                    'theta_bin': theta_bin, 'r_bin': r_bin, 'w_bin': w_bin, 'x_bin': x_bin, 'y_bin': y_bin, 'weight_bin': hist, 'r_sig_bin': w_bin/np.sqrt(hist),
                    'r_thresh': r_thresh, 'N_thresh': N_thresh, 'n_steps':list_n_steps, 'xv_stream': xv_stream, 'xv_sat': xv_sat_forward}
    
    return dict_stream
