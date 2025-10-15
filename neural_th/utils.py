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
        temp_down = mvpa.TEMP_mvp[down_idx]
        cond_down = mvpa.COND_mvp[down_idx] 
        pres_down = mvpa.PRES_mvp[down_idx]
        speed_down = mvpa.SPEED_mvp[down_idx]
        
        temp_up = mvpa.TEMP_mvp[up_idx]
        cond_up = mvpa.COND_mvp[up_idx]
        pres_up = mvpa.PRES_mvp[up_idx]
        speed_up = mvpa.SPEED_mvp[up_idx]
        
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
            salt_down_ref = mvpa.SALT_ctd[down_idx]
            salt_up_ref = mvpa.SALT_ctd[up_idx]
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