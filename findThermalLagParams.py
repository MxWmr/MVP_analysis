"""
Thermal lag parameter estimation for CTD profiles.

This module estimates thermal lag parameters for CTD (Conductivity-Temperature-Depth)
profiles by minimizing the area between two profiles in a Temperature-Salinity diagram.

References:
    Garau et al. (2011): Thermal Lag Correction on Slocum CTD Glider Data.
    Morison et al. (1994): The Correction for Thermal-Lag Effects in Sea-Bird CTD Data.
    Lueck & Picklo (1990): Thermal Inertia of Conductivity Cells.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import gsw  # Gibbs SeaWater package for oceanographic calculations


def find_thermal_lag_params(
    time1: np.ndarray,
    cond1: np.ndarray,
    temp1: np.ndarray,
    pres1: np.ndarray,
    time2: np.ndarray,
    cond2: np.ndarray,
    temp2: np.ndarray,
    pres2: np.ndarray,
    flow1: Optional[np.ndarray] = None,
    flow2: Optional[np.ndarray] = None,
    graphics: bool = False,
    guess: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    optim_opts: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Find thermal lag parameters for CTD profiles.
    
    Parameters
    ----------
    time1, time2 : np.ndarray
        Time vectors in seconds for profiles 1 and 2
    cond1, cond2 : np.ndarray
        Conductivity vectors in S/m for profiles 1 and 2
    temp1, temp2 : np.ndarray
        Temperature vectors in °C for profiles 1 and 2
    pres1, pres2 : np.ndarray
        Pressure vectors in dbar for profiles 1 and 2
    flow1, flow2 : np.ndarray, optional
        Flow speed vectors in m/s for unpumped CTD (variable flow speed)
        If None, assumes constant flow speed (pumped CTD)
    graphics : bool, default=False
        If True, display optimization progress plots
    guess : np.ndarray, optional
        Initial parameter guess. Default values used if None
    lower : np.ndarray, optional
        Lower bounds for parameters
    upper : np.ndarray, optional
        Upper bounds for parameters
    optim_opts : dict, optional
        Additional options for scipy.optimize.minimize
        
    Returns
    -------
    params : np.ndarray
        Estimated thermal lag parameters
        - For constant flow (pumped CTD): [alpha, tau]
        - For variable flow (unpumped CTD): [alpha_o, alpha_s, tau_o, tau_s]
    exitflag : int
        Optimization exit status (>0 for success)
    residual : float
        Residual area between corrected profiles
        
    Examples
    --------
    >>> # Constant flow speed (pumped CTD)
    >>> params, exitflag, residual = find_thermal_lag_params(
    ...     time1, cond1, temp1, pres1,
    ...     time2, cond2, temp2, pres2
    ... )
    
    >>> # Variable flow speed (unpumped CTD)
    >>> params, exitflag, residual = find_thermal_lag_params(
    ...     time1, cond1, temp1, pres1, flow1,
    ...     time2, cond2, temp2, pres2, flow2,
    ...     graphics=True
    ... )
    """
    
    # Determine if constant or variable flow
    constant_flow = (flow1 is None and flow2 is None)
    
    # Configure default options
    time_range = min(np.ptp(time1), np.ptp(time2))
    
    if constant_flow:
        # Default for pumped CTD (constant flow ~0.4867 m/s)
        default_guess = np.array([0.0677, 11.1431])
        default_lower = np.array([0.0, 0.0])
        default_upper = np.array([4.0, 2.5 * time_range])
    else:
        # Default for unpumped CTD from Morison (1994)
        default_guess = np.array([0.0135, 0.0264, 7.1499, 2.7858])
        default_lower = np.array([0.0, 0.0, 0.0, 0.0])
        default_upper = np.array([2.0, 1.0, time_range, time_range / 2])
    
    # Use provided values or defaults
    guess = default_guess if guess is None else np.asarray(guess)
    lower = default_lower if lower is None else np.asarray(lower)
    upper = default_upper if upper is None else np.asarray(upper)
    
    # Default optimization options - using L-BFGS-B for better compatibility
    default_optim_opts = {
        'method': 'L-BFGS-B',
        'options': {
            'ftol': 1e-4,
            'gtol': 1e-5,
            'maxiter': 1000,
            'disp': graphics
        }
    }
    
    if optim_opts is not None:
        default_optim_opts.update(optim_opts)
    
    # Setup for graphics if requested
    if graphics:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Thermal Lag Parameter Optimization', fontsize=14)
        plt.ion()
        plt.show()
        
        # Store data for plotting callbacks
        plot_data = {
            'fig': fig,
            'axes': axes.flatten(),
            'iteration': 0,
            'params_history': [],
            'residual_history': []
        }
    else:
        plot_data = None
    
    # Define objective function
    def objective(params):
        """Compute area enclosed by profiles in T-S diagram."""
        try:
            # Correct thermal lag with safety checks
            if constant_flow:
                temp_cor1, _ = correct_thermal_lag(time1, cond1, temp1, params)
                temp_cor2, _ = correct_thermal_lag(time2, cond2, temp2, params)
            else:
                temp_cor1, _ = correct_thermal_lag(time1, cond1, temp1, params, flow1)
                temp_cor2, _ = correct_thermal_lag(time2, cond2, temp2, params, flow2)
            print(np.sum(np.isnan(temp_cor1)), np.sum(np.isnan(temp_cor2)), len(temp_cor1))
            # Safety check for temperature values
            if (temp_cor1 is None or temp_cor2 is None or 
                not np.any(np.isfinite(temp_cor1)) or not np.any(np.isfinite(temp_cor2)) or
                np.any(temp_cor1 < -10) or np.any(temp_cor1 > 50) or
                np.any(temp_cor2 < -10) or np.any(temp_cor2 > 50)):
                return 1e6  # Pénalité pour températures aberrantes
                
        except Exception as e:
            return 1e6  # Pénalité en cas d'erreur
        
        # Calculate practical salinity
        salt_cor1 = gsw.SP_from_C(cond1, temp_cor1, pres1)
        salt_cor2 = gsw.SP_from_C(cond2, temp_cor2, pres2)
        
        # Calculate density
        # SA1 = gsw.SA_from_SP(salt_cor1, pres1, 0, 0)  # Absolute Salinity
        # SA2 = gsw.SA_from_SP(salt_cor2, pres2, 0, 0)
        # CT1 = gsw.CT_from_t(SA1, temp_cor1, pres1)  # Conservative Temperature
        # CT2 = gsw.CT_from_t(SA2, temp_cor2, pres2)
        # dens_cor1 = gsw.rho(SA1, CT1, pres1)
        # dens_cor2 = gsw.rho(SA2, CT2, pres2)
        
        # # Find common density range
        # dens_min = max(np.nanmin(dens_cor1), np.nanmin(dens_cor2))
        # dens_max = min(np.nanmax(dens_cor1), np.nanmax(dens_cor2))
        
        # # Mask data within common density range
        # mask1 = (dens_cor1 >= dens_min) & (dens_cor1 <= dens_max)
        # mask2 = (dens_cor2 >= dens_min) & (dens_cor2 <= dens_max)
        # print(mask1.sum(), mask2.sum(),len(mask1))


        mask1 = ~np.isnan(salt_cor1) & ~np.isnan(temp_cor1)
        mask2 = ~np.isnan(salt_cor2) & ~np.isnan(temp_cor2)
        # Calculate area between profiles
        area = profile_area(
            salt_cor1[mask1], temp_cor1[mask1],
            salt_cor2[mask2], temp_cor2[mask2]
        )
        
        # Update plots if graphics enabled
        if plot_data is not None:
            plot_data['iteration'] += 1
            plot_data['params_history'].append(params.copy())
            plot_data['residual_history'].append(area)
            
            if plot_data['iteration'] % 5 == 0:  # Update every 5 iterations
                update_plots(plot_data, params, area, constant_flow,
                           time1, cond1, temp1, pres1, flow1,
                           time2, cond2, temp2, pres2, flow2)
        
        return area
    
    # Define bounds
    bounds = list(zip(lower, upper))
    
    # Run optimization
    result = minimize(
        objective,
        guess,
        bounds=bounds,
        **default_optim_opts
    )
    
    params = result.x
    exitflag = 1 if result.success else 0
    residual = result.fun
    
    if graphics:
        plt.ioff()
        print(f"\nOptimization completed:")
        print(f"  Parameters: {params}")
        print(f"  Residual: {residual:.6f}")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
    
    return params, exitflag, residual






"""
Correct CTD conductivity and temperature sequences from thermal lag effects.

This module implements thermal lag correction for CTD data based on the 
recursive algorithm described in Lueck (1990), Morison (1994), and Garau (2011).

References:
    Garau et al. (2011): Thermal Lag Correction on Slocum CTD Glider Data.
    Morison et al. (1994): The Correction for Thermal-Lag Effects in Sea-Bird CTD Data.
    Lueck & Picklo (1990): Thermal Inertia of Conductivity Cells.
"""

import numpy as np
from typing import Tuple, Optional


def correct_thermal_lag(
    timestamp: np.ndarray,
    cond_inside: np.ndarray,
    temp_outside: np.ndarray,
    params: np.ndarray,
    flow_speed: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct CTD conductivity and temperature from thermal lag effects.
    
    The thermal lag correction accounts for the slower thermal response of the
    conductivity cell compared to the temperature sensor. This creates an error
    when the CTD encounters rapid temperature changes.
    
    Parameters
    ----------
    timestamp : np.ndarray
        Sample timestamps in seconds (shape: N,)
    cond_inside : np.ndarray
        Measured conductivity inside CTD cell in S/m (shape: N,)
    temp_outside : np.ndarray
        Measured temperature outside CTD cell in °C (shape: N,)
    params : np.ndarray
        Correction parameters:
        - For constant flow (pumped CTD): [alpha, tau]
          * alpha: error magnitude (dimensionless)
          * tau: error time constant (seconds)
        - For variable flow (unpumped CTD): [alpha_o, alpha_s, tau_o, tau_s]
          * alpha_o: error magnitude offset
          * alpha_s: error magnitude slope (m/s)
          * tau_o: error time constant offset (s)
          * tau_s: error time constant slope (s·√(m/s))
    flow_speed : np.ndarray, optional
        Flow speed through CTD cell in m/s (shape: N,)
        Required for variable flow speed (unpumped CTD)
        If None, assumes constant flow speed (pumped CTD)
        
    Returns
    -------
    temp_inside : np.ndarray
        Corrected temperature inside CTD cell in °C (shape: N,)
    cond_outside : np.ndarray
        Corrected conductivity outside CTD cell in S/m (shape: N,)
        
    Notes
    -----
    The correction uses a recursive algorithm that requires sequential processing.
    Invalid data points (NaN or timestamp <= 0) are preserved as NaN in output.
    
    The conductivity sensitivity to temperature is approximated as:
        dC/dT ≈ 0.088 + 0.0006 * T
    
    This is based on SeaBird's approximation for typical seawater conditions.
    
    Examples
    --------
    >>> # Pumped CTD (constant flow speed)
    >>> params = np.array([0.0677, 11.1431])  # alpha, tau
    >>> temp_corrected, cond_corrected = correct_thermal_lag(
    ...     timestamp, cond_measured, temp_measured, params
    ... )
    
    >>> # Unpumped CTD (variable flow speed)
    >>> params = np.array([0.0135, 0.0264, 7.1499, 2.7858])
    >>> temp_corrected, cond_corrected = correct_thermal_lag(
    ...     timestamp, cond_measured, temp_measured, params, flow_speed
    ... )
    
    See Also
    --------
    find_thermal_lag_params : Estimate thermal lag parameters from profile pairs
    compute_ctd_flow_speed : Calculate flow speed for unpumped CTD
    """
    
    # Ensure inputs are numpy arrays
    timestamp = np.asarray(timestamp).flatten()
    cond_inside = np.asarray(cond_inside).flatten()
    temp_outside = np.asarray(temp_outside).flatten()
    params = np.asarray(params)
    
    # Validate input dimensions
    n_samples = len(timestamp)
    if len(cond_inside) != n_samples or len(temp_outside) != n_samples:
        raise ValueError("timestamp, cond_inside, and temp_outside must have same length")
    
    # Determine if constant or variable flow
    constant_flow = (flow_speed is None)
    
    # Select valid data points
    if constant_flow:
        valid = (
            (timestamp > 0) &
            ~np.isnan(cond_inside) &
            ~np.isnan(temp_outside)
        )
    else:
        flow_speed = np.asarray(flow_speed).flatten()
        if len(flow_speed) != n_samples:
            raise ValueError("flow_speed must have same length as timestamp")
        
        valid = (
            (timestamp > 0) &
            ~np.isnan(cond_inside) &
            ~np.isnan(temp_outside) &
            ~np.isnan(flow_speed)
        )
    
    # Extract valid values
    time_val = timestamp[valid]
    temp_val = temp_outside[valid]
    cond_val = cond_inside[valid]
    
    n_valid = len(time_val)
    
    # Handle edge case: no valid data
    if n_valid == 0:
        return np.full_like(timestamp, np.nan), np.full_like(timestamp, np.nan)
    
    # Extract and compute thermal lag parameters
    if constant_flow:
        # Constant flow speed (pumped CTD)
        if len(params) != 2:
            raise ValueError("For constant flow, params must be [alpha, tau]")
        
        alpha = params[0]
        tau = params[1]
        
        # Create arrays of constant parameters for all time steps
        alpha_vec = np.full(n_valid - 1, alpha)
        tau_vec = np.full(n_valid - 1, tau)
        
    else:
        # Variable flow speed (unpumped CTD)
        if len(params) != 4:
            raise ValueError("For variable flow, params must be [alpha_o, alpha_s, tau_o, tau_s]")
        
        alpha_offset, alpha_slope, tau_offset, tau_slope = params
        flow_val = flow_speed[valid]
        
        # Compute dynamic parameters at each time step (except last)
        # Use flow at current time step for correction to next step
        alpha_vec = alpha_offset + alpha_slope / flow_val[:-1]
        tau_vec = tau_offset + tau_slope / np.sqrt(flow_val[:-1])
    
    # Compute time differences between consecutive samples
    dtime = np.diff(time_val)
    
    # Compute correction coefficients
    # Based on Lueck (1990) and Morison (1994) formulations
    coef_a = 2 * alpha_vec / (2 + dtime / tau_vec)
    coef_b = 1 - 4 / (2 + dtime / tau_vec)
    
    # Compute conductivity sensitivity to temperature
    # SeaBird approximation: dC/dT ≈ 0.088 + 0.0006 * T
    dcond_dtemp = 0.088 + 0.0006 * temp_val
    
    # Compute temperature differences between consecutive samples
    dtemp = np.diff(temp_val)
    
    # Initialize correction vectors
    cond_correction = np.zeros(n_valid)
    temp_correction = np.zeros(n_valid)
    
    # Apply recursive correction formula
    # This loop cannot be easily vectorized due to recursive dependency
    for n in range(n_valid - 1):
        # Correction for conductivity at next time step
        cond_correction[n + 1] = (
            -coef_b[n] * cond_correction[n] +
            coef_a[n] * dcond_dtemp[n] * dtemp[n]
        )
        
        # Correction for temperature at next time step
        temp_correction[n + 1] = (
            -coef_b[n] * temp_correction[n] +
            coef_a[n] * dtemp[n]
        )
    
    # Apply corrections to valid data
    # Temperature inside = measured temperature outside - correction
    # Conductivity outside = measured conductivity inside + correction
    temp_inside_val = temp_val - temp_correction
    cond_outside_val = cond_val + cond_correction
    
    # Create output arrays with NaN for invalid points
    temp_inside = np.full(n_samples, np.nan)
    cond_outside = np.full(n_samples, np.nan)
    
    # Fill in corrected values at valid positions
    temp_inside[valid] = temp_inside_val
    cond_outside[valid] = cond_outside_val
    
    return temp_inside, cond_outside


def apply_thermal_lag_correction_batch(
    timestamps: list,
    conds_inside: list,
    temps_outside: list,
    params: np.ndarray,
    flow_speeds: Optional[list] = None
) -> Tuple[list, list]:
    """
    Apply thermal lag correction to multiple profiles.
    
    Convenience function to process multiple CTD profiles with the same
    correction parameters.
    
    Parameters
    ----------
    timestamps : list of np.ndarray
        List of timestamp arrays for each profile
    conds_inside : list of np.ndarray
        List of conductivity arrays for each profile
    temps_outside : list of np.ndarray
        List of temperature arrays for each profile
    params : np.ndarray
        Thermal lag correction parameters (same for all profiles)
    flow_speeds : list of np.ndarray, optional
        List of flow speed arrays for each profile (for unpumped CTD)
        
    Returns
    -------
    temps_inside : list of np.ndarray
        List of corrected temperature arrays
    conds_outside : list of np.ndarray
        List of corrected conductivity arrays
        
    Examples
    --------
    >>> # Process multiple profiles
    >>> params = np.array([0.0135, 0.0264, 7.1499, 2.7858])
    >>> temps_corrected, conds_corrected = apply_thermal_lag_correction_batch(
    ...     [time1, time2, time3],
    ...     [cond1, cond2, cond3],
    ...     [temp1, temp2, temp3],
    ...     params,
    ...     [flow1, flow2, flow3]
    ... )
    """
    n_profiles = len(timestamps)
    
    if flow_speeds is None:
        flow_speeds = [None] * n_profiles
    
    temps_inside = []
    conds_outside = []
    
    for i in range(n_profiles):
        temp_in, cond_out = correct_thermal_lag(
            timestamps[i],
            conds_inside[i],
            temps_outside[i],
            params,
            flow_speeds[i]
        )
        temps_inside.append(temp_in)
        conds_outside.append(cond_out)
    
    return temps_inside, conds_outside


def compute_ctd_flow_speed(
    timestamp: np.ndarray,
    depth: np.ndarray,
    pitch: np.ndarray,
    min_pitch: float = np.deg2rad(11)
) -> np.ndarray:
    """
    Compute flow speed through CTD cell for unpumped CTD.
    
    For unpumped CTD (e.g., on gliders), the flow speed depends on the
    vehicle's vertical velocity and pitch angle.
    
    Parameters
    ----------
    timestamp : np.ndarray
        Time in seconds (shape: N,)
    depth : np.ndarray
        Depth in meters (shape: N,)
    pitch : np.ndarray
        Pitch angle in radians (positive = nose up) (shape: N,)
    min_pitch : float, default=11°
        Minimum pitch angle (in radians) to consider for flow calculation
        Below this angle, use vertical velocity directly
        
    Returns
    -------
    flow_speed : np.ndarray
        Flow speed through CTD in m/s (shape: N,)
        
    Notes
    -----
    Flow speed is approximated as:
        flow ≈ |dz/dt| / |sin(pitch)|
    
    For small pitch angles (< min_pitch), the vertical velocity is used
    directly to avoid numerical instabilities.
    
    Flow speeds are clipped to [0.01, 2.0] m/s to avoid unrealistic values.
    
    Examples
    --------
    >>> flow = compute_ctd_flow_speed(time, depth, pitch)
    >>> # Use in thermal lag correction
    >>> temp_cor, cond_cor = correct_thermal_lag(
    ...     time, cond, temp, params, flow
    ... )
    """
    # Compute vertical velocity (dz/dt)
    dz_dt = np.gradient(depth, timestamp)
    
    # Compute flow speed: vertical velocity / sin(pitch)
    flow_speed = np.abs(dz_dt / np.sin(pitch))
    
    # For small pitch angles, use vertical velocity directly
    small_pitch = np.abs(pitch) < min_pitch
    flow_speed[small_pitch] = np.abs(dz_dt[small_pitch])
    
    # Clip to reasonable range (1 cm/s to 2 m/s)
    flow_speed = np.clip(flow_speed, 0.01, 2.0)
    
    return flow_speed




def profile_area(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray
) -> float:
    """
    Calculate area enclosed by two profiles with opposite directions.
    
    This function computes the area enclosed by two profiles by joining them
    into a polygon and decomposing it into triangles. This handles complex
    (self-intersecting) polygons correctly.
    
    Parameters
    ----------
    x1, y1 : np.ndarray
        Coordinates of first profile
    x2, y2 : np.ndarray
        Coordinates of second profile (opposite direction)
        
    Returns
    -------
    area : float
        Total area enclosed by the two profiles
        
    Notes
    -----
    Invalid coordinates (NaN) are automatically removed before computation.
    The polygon formed by joining both profiles may be self-intersecting,
    so the area is computed by triangulation.
    
    Examples
    --------
    >>> x1 = np.array([0, -1, 1, 0])
    >>> y1 = np.array([2, 1, -1, -2])
    >>> x2 = np.array([0, -1, 1, 0])
    >>> y2 = np.array([-2, -1, 1, 2])
    >>> area = profile_area(x1, y1, x2, y2)
    """
    from matplotlib.tri import Triangulation
    from shapely.geometry import Polygon
    
    # Flatten and concatenate both profiles
    x1 = np.asarray(x1).flatten()
    y1 = np.asarray(y1).flatten()
    x2 = np.asarray(x2).flatten()
    y2 = np.asarray(y2).flatten()
    
    # Join both profiles
    x_all = np.concatenate([x1, x2])
    y_all = np.concatenate([y1, y2])
    
    # Remove NaN values
    valid = ~(np.isnan(x_all) | np.isnan(y_all))
    x_clean = x_all[valid]
    y_clean = y_all[valid]
    
    # Need at least 3 points to form a polygon
    if len(x_clean) < 3:
        return np.inf
    
    try:
        # Create polygon from the joined profiles
        # Close the polygon if not already closed
        if not (x_clean[0] == x_clean[-1] and y_clean[0] == y_clean[-1]):
            x_poly = np.append(x_clean, x_clean[0])
            y_poly = np.append(y_clean, y_clean[0])
        else:
            x_poly = x_clean
            y_poly = y_clean
        
        # Use Shapely for robust polygon area calculation
        # It handles self-intersecting polygons correctly
        points = list(zip(x_poly, y_poly))
        polygon = Polygon(points)
        
        # Get area (absolute value for potentially complex polygons)
        area = abs(polygon.area)
        
        # If polygon is invalid (self-intersecting), use triangulation method
        if not polygon.is_valid or np.isinf(area) or np.isnan(area):
            area = _triangulation_area(x_clean, y_clean)
            
    except Exception:
        # Fallback to triangulation method if Shapely fails
        area = _triangulation_area(x_clean, y_clean)
    
    return area


def _triangulation_area(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute polygon area using triangulation decomposition.
    
    Helper function that decomposes a complex polygon into triangles
    and sums their areas. This handles self-intersecting polygons.
    """
    from matplotlib.tri import Triangulation
    from matplotlib.path import Path
    
    try:
        # Create Delaunay triangulation
        tri = Triangulation(x, y)
        
        # Compute area of each triangle and sum
        triangles = tri.triangles
        total_area = 0.0
        
        for triangle in triangles:
            # Get vertices of triangle
            x_tri = x[triangle]
            y_tri = y[triangle]
            
            # Compute triangle area using cross product formula
            # Area = 0.5 * |cross product|
            tri_area = 0.5 * abs(
                (x_tri[1] - x_tri[0]) * (y_tri[2] - y_tri[0]) -
                (x_tri[2] - x_tri[0]) * (y_tri[1] - y_tri[0])
            )
            total_area += tri_area
        
        return total_area
        
    except Exception:
        # If triangulation fails, return infinity
        return np.inf

def update_plots(
    plot_data: Dict,
    params: np.ndarray,
    residual: float,
    constant_flow: bool,
    time1, cond1, temp1, pres1, flow1,
    time2, cond2, temp2, pres2, flow2
):
    """Update optimization progress plots."""
    axes = plot_data['axes']
    
    # Apply correction with current parameters
    if constant_flow:
        temp_cor1 = correct_thermal_lag(time1, cond1, temp1, params)
        temp_cor2 = correct_thermal_lag(time2, cond2, temp2, params)
    else:
        temp_cor1 = correct_thermal_lag(time1, cond1, temp1, params, flow1)
        temp_cor2 = correct_thermal_lag(time2, cond2, temp2, params, flow2)
    
    salt_cor1 = gsw.SP_from_C(cond1 , temp_cor1, pres1)
    salt_cor2 = gsw.SP_from_C(cond2 , temp_cor2, pres2)
    salt1 = gsw.SP_from_C(cond1 , temp1, pres1)
    salt2 = gsw.SP_from_C(cond2 , temp2, pres2)
    
    # Clear all axes
    for ax in axes:
        ax.clear()
    
    # Plot 1: Parameter evolution
    if len(plot_data['params_history']) > 1:
        params_array = np.array(plot_data['params_history'])
        for i in range(params_array.shape[1]):
            axes[0].plot(params_array[:, i], label=f'param_{i}')
        axes[0].set_title('Parameter Evolution')
        axes[0].set_xlabel('Iteration')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot 2: Residual evolution
    if len(plot_data['residual_history']) > 1:
        axes[1].plot(plot_data['residual_history'])
        axes[1].set_title(f'Residual: {residual:.6f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_yscale('log')
        axes[1].grid(True)
    
    # Plot 3: T-S diagram
    axes[2].plot(salt1, temp1, ':r', alpha=0.5, label='Profile 1 original')
    axes[2].plot(salt2, temp2, ':b', alpha=0.5, label='Profile 2 original')
    axes[2].plot(salt_cor1, temp_cor1, '-r', label='Profile 1 corrected')
    axes[2].plot(salt_cor2, temp_cor2, '-b', label='Profile 2 corrected')
    axes[2].set_title('Temperature-Salinity')
    axes[2].set_xlabel('Salinity (PSU)')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: P-S diagram
    axes[3].plot(salt1, pres1, ':r', alpha=0.5)
    axes[3].plot(salt2, pres2, ':b', alpha=0.5)
    axes[3].plot(salt_cor1, pres1, '-r')
    axes[3].plot(salt_cor2, pres2, '-b')
    axes[3].set_title('Pressure-Salinity')
    axes[3].set_xlabel('Salinity (PSU)')
    axes[3].set_ylabel('Pressure (dbar)')
    axes[3].invert_yaxis()
    axes[3].grid(True)
    
    # Plot 5: T-time diagram
    time_offset = min(time1.min(), time2.min())
    axes[4].plot(time1 - time_offset, temp1, ':r', alpha=0.5)
    axes[4].plot(time2 - time_offset, temp2, ':b', alpha=0.5)
    axes[4].plot(time1 - time_offset, temp_cor1, '-r')
    axes[4].plot(time2 - time_offset, temp_cor2, '-b')
    axes[4].set_title('Temperature-Time')
    axes[4].set_xlabel('Time (s)')
    axes[4].set_ylabel('Temperature (°C)')
    axes[4].grid(True)
    
    # Plot 6: C-time diagram
    axes[5].plot(time1 - time_offset, cond1, ':r', alpha=0.5)
    axes[5].plot(time2 - time_offset, cond2, ':b', alpha=0.5)
    axes[5].set_title('Conductivity-Time')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Conductivity (S/m)')
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.pause(0.01)


if __name__ == "__main__":
    # Example usage
    print("Thermal Lag Parameter Estimation")
    print("================================")
    print("This module requires input data to run.")
    print("See function docstring for usage examples.")