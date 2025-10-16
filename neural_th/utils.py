# Prepare data for neural network training
import gsw
import numpy as np

def prepare_training_data(mvpa, min_points=500):
    """
    Prepare MVP and CTD data for neural network training
    """
    mvp_data = {
        'TEMP_down': [],
        'COND_down': [],
        'PRES_down': [],
        'SPEED_down': [],
        'TEMP_up': [],
        'COND_up': [], 
        'PRES_up': [],
        'SPEED_up': []
    }
    
    ctd_data = {
        'SALT_down': [],
        'SALT_up': []
    }
    
    # Group profiles into down/up pairs
    n_profiles = len(mvpa.PRES_mvp)
    print(f"Processing {n_profiles} profiles...")
    
    valid_pairs = 0
    
    # Assume profiles are paired: 0,1 = pair 1; 2,3 = pair 2; etc.
    for i in range(0, n_profiles-1, 2):
        down_idx = i
        up_idx = i + 1
        
        # Check data quality
        temp_down = mvpa.TEMP_mvp_corr_interp[down_idx]
        cond_down = mvpa.COND_mvp_corr_interp[down_idx] 
        pres_down = mvpa.PRES_mvp_corr_interp[down_idx]
        speed_down = mvpa.SPEED_mvp_corr_interp[down_idx]
        
        temp_up = mvpa.TEMP_mvp_corr_interp[up_idx]
        cond_up = mvpa.COND_mvp_corr_interp[up_idx]
        pres_up = mvpa.PRES_mvp_corr_interp[up_idx]
        speed_up = mvpa.SPEED_mvp_corr_interp[up_idx]
        
        # Check if profiles have enough valid data
        valid_down = (~np.isnan(temp_down)) & (~np.isnan(cond_down)) & (~np.isnan(pres_down)) & (~np.isnan(speed_down))
        valid_up = (~np.isnan(temp_up)) & (~np.isnan(cond_up)) & (~np.isnan(pres_up)) & (~np.isnan(speed_up))
        
        n_valid_down = np.sum(valid_down)
        n_valid_up = np.sum(valid_up)
        
        if n_valid_down < min_points or n_valid_up < min_points:
            print(f"Skipping pair ({down_idx},{up_idx}): insufficient data ({n_valid_down}, {n_valid_up})")
            continue
        
        # Add MVP data
        mvp_data['TEMP_down'].append(temp_down)
        mvp_data['COND_down'].append(cond_down)
        mvp_data['PRES_down'].append(pres_down)
        mvp_data['SPEED_down'].append(speed_down)
        
        mvp_data['TEMP_up'].append(temp_up)
        mvp_data['COND_up'].append(cond_up)
        mvp_data['PRES_up'].append(pres_up)
        mvp_data['SPEED_up'].append(speed_up)
        
        # Get or compute CTD reference salinity
        if hasattr(mvpa, 'SALT_ctd') and len(mvpa.SALT_ctd) > up_idx:
            # Use actual CTD data
            salt_down_ref = mvpa.SALT_ctd_on_mvp[down_idx]
            salt_up_ref = mvpa.SALT_ctd_on_mvp[up_idx]
        else:
            # Compute reference salinity from MVP data (for demonstration)
            print(f"Computing reference salinity for pair ({down_idx},{up_idx})")
            
            # Use only valid points for salinity computation
            salt_down_ref = np.full_like(temp_down, np.nan)
            salt_up_ref = np.full_like(temp_up, np.nan)
            
            if np.sum(valid_down) > 0:
                salt_vals = gsw.SP_from_C(cond_down[valid_down], 
                                        temp_down[valid_down], 
                                        pres_down[valid_down])
                salt_down_ref[valid_down] = salt_vals
                
            if np.sum(valid_up) > 0:
                salt_vals = gsw.SP_from_C(cond_up[valid_up], 
                                        temp_up[valid_up], 
                                        pres_up[valid_up])
                salt_up_ref[valid_up] = salt_vals
        
        ctd_data['SALT_down'].append(salt_down_ref)
        ctd_data['SALT_up'].append(salt_up_ref)
        
        valid_pairs += 1
        print(f"Added pair ({down_idx},{up_idx}) - Down: {n_valid_down} pts, Up: {n_valid_up} pts")
    
    print(f"\nPrepared {valid_pairs} valid profile pairs for training")
    return mvp_data, ctd_data



def garau_correction_nograd(T, C, P, V, alpha0, alphaS, tau0, tauS, fs=20):
    """
    Apply Garau correction (COMPLETELY SAFE VERSION - no loops, no in-place)
    """
    batch_size, seq_len = T.shape
    device = T.device
    
    # Use mean flow speed for each profile
    V_mean = torch.mean(V, dim=1)  # [batch_size]
    
    # Compute coefficients with aggressive clamping
    alpha = torch.clamp(alpha0 + alphaS * V_mean, min=1e-3, max=0.5)  # Smaller range
    tau = torch.clamp(tau0 + tauS * V_mean, min=1.0, max=50.0)        # Smaller range
    
    # Compute correction coefficients with safety
    denominator = torch.clamp(1 + 4 * fs * tau, min=1e-3)
    a = torch.clamp(4 * fs * alpha * tau / denominator, min=1e-6, max=1.0)
    
    # Avoid division by alpha - use alternative formulation
    # Original: b = 1 - 2*a/alpha  ← PROBLÉMATIQUE
    # Alternative: b = (1 + 4*fs*tau - 2*4*fs*tau) / (1 + 4*fs*tau)
    b = torch.clamp((1 - 8 * fs * tau) / denominator, min=-5.0, max=5.0)
    
    # Expand for broadcasting
    a_exp = a.unsqueeze(1)  # [batch_size, 1]
    b_exp = b.unsqueeze(1)  # [batch_size, 1]
    
    # Compute temperature differences (vectorized)
    dT = torch.zeros_like(T)
    dT[:, 1:] = T[:, 1:] - T[:, :-1]
    
    # SIMPLIFIED NON-RECURSIVE CORRECTION (no loops!)
    # Instead of recursive formula, use first-order approximation
    correction_factor = a_exp * tau.unsqueeze(1)  # Combined correction
    T_corrected = T + correction_factor * dT
    
    # Ensure output is reasonable
    T_corrected = torch.clamp(T_corrected, min=T.min()-5.0, max=T.max()+5.0)
    
    return T


def garau_correction_nograd(T, C, P, V, alpha0, alphaS, tau0, tauS, fs=20):
    """
    Apply Garau correction using NumPy (for 1D arrays)
    
    Args:
        T: Temperature array (1D)
        C: Conductivity array (1D)
        P: Pressure array (1D)
        V: Flow speed array (1D)
        alpha0: Amplitude parameter at surface (scalar)
        alphaS: Salinity dependence of amplitude (scalar)
        tau0: Time constant at surface (scalar)
        tauS: Salinity dependence of time constant (scalar)
        fs: Sampling frequency (default: 20 Hz)
    
    Returns:
        T_corrected: Corrected temperature array (1D)
    """
    # Use mean flow speed
    V_mean = np.nanmean(V)
    
    # Compute coefficients with clamping
    alpha = np.clip(alpha0 + alphaS * V_mean, 1e-3, 0.5)
    tau = np.clip(tau0 + tauS * V_mean, 1.0, 50.0)
    
    # Compute correction coefficients
    denominator = np.clip(1 + 4 * fs * tau, 1e-3, None)
    a = np.clip(4 * fs * alpha * tau / denominator, 1e-6, 1.0)
    
    # Alternative formulation to avoid division by alpha
    b = np.clip((1 - 8 * fs * tau) / denominator, -5.0, 5.0)
    
    # Compute temperature differences
    dT = np.zeros_like(T)
    dT[1:] = T[1:] - T[:-1]
    
    # Apply correction (first-order approximation)
    correction_factor = a * tau
    T_corrected = T + correction_factor * dT
    
    # Clamp to reasonable range
    T_min = np.nanmin(T) - 5.0
    T_max = np.nanmax(T) + 5.0
    T_corrected = np.clip(T_corrected, T_min, T_max)
    
    return T_corrected

