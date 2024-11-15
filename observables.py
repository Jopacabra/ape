import numpy as np
import xarray as xr
import timeit


# Function to compute vns for a grouping of hard particles in an xarray dataarray
def compute_vns(xr_da, n_list=np.array([2, 3, 4])):
    ###################
    # Compute results #
    ###################
    seed = int(xr_da.attrs['seed'])
    # Compute v2
    t_0 = timeit.default_timer()
    # First we need to compute the pt and phi positions of all of the particles.
    phis = xr_da.phi_f.to_numpy()
    pts = xr_da.pt_f.to_numpy()

    # Create dictionaries for per-event usage
    event_hard_v = {}
    event_hard_psi = {}

    # Reset event arrays
    for n in n_list:
        event_hard_v[n] = np.array([])
        event_hard_psi[n] = np.array([])

    event_weight = np.array([])
    for pt in pts:
        # Get the weight in this pt bin, summed over all ids
        weights = xr_da.sel(pt_f=pt).to_numpy()
        event_weight = np.append(event_weight, np.sum(weights))
        for n in n_list:
            # Compute hard psi_2
            sin_phases = np.sum(np.sin(n * phis) * weights)
            cos_phases = np.sum(np.cos(n * phis) * weights)
            hard_psi_n = (1 / n) * np.arctan2(sin_phases, cos_phases)

            # Compute the v2 phase associated with each phi position
            phases = np.cos(n * (phis - hard_psi_n))

            # Compute v2 and append to storage array -- divide by zero gives NaN for numpy array sums like this
            hard_vn = np.sum(weights * phases) / np.sum(weights)

            event_hard_v[n] = np.append(event_hard_v[n], hard_vn)
            event_hard_psi[n] = np.append(event_hard_psi[n], hard_psi_n)

    # Record vn and psin
    event_hard_vn_stacked = np.stack([event_hard_v[n] for n in n_list], axis=0)
    event_hard_psin_stacked = np.stack([event_hard_psi[n] for n in n_list], axis=0)
    vn_dataarray = xr.DataArray(event_hard_vn_stacked, coords={"n": n_list, "pt": pts})
    psin_dataarray = xr.DataArray(event_hard_psin_stacked, coords={"n": n_list, "pt": pts})
    weight_dataarray = xr.DataArray(event_weight, coords={"pt": pts})

    # Apply attributes to output dataarray
    for output_da in [vn_dataarray, psin_dataarray, weight_dataarray]:
        output_da.attrs = xr_da.attrs

    # Add arrays to proper dataframe
    result_ds = xr.Dataset({})
    result_ds[str(seed) + "_vn"] = vn_dataarray
    result_ds[str(seed) + "_psin"] = psin_dataarray
    result_ds[str(seed) + "_weight"] = weight_dataarray

    return result_ds


# Function to compute R_AA for a grouping of hard particles in two xarray dataarrays
def compute_raa(xr_da_f, xr_da_i):
    seed = int(xr_da_f.attrs['seed'])

    # Compute R_AA for each pt bin
    pts = xr_da_f.pt.to_numpy()  # Get pt bins
    event_raa = np.array([])
    for pt in pts:
        # Get the weight in this pt bin, summed over all ids
        weight_f = xr_da_f.sel(pt=pt).sum()
        weight_i = xr_da_i.sel(pt=pt).sum()
        try:
            raa = weight_f / weight_i
        except:
            raa = np.nan
        event_raa = np.append(event_raa, raa)

    # Create xarray dataarray
    raa_dataarray = xr.DataArray(event_raa, coords={"pt": pts})

    # Apply attributes to output dataarray
    raa_dataarray.attrs = xr_da_f.attrs

    # Add arrays to proper dataframe
    result_ds = xr.Dataset({})
    result_ds[str(seed) + "_raa"] = raa_dataarray

    return result_ds