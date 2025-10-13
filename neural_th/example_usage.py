"""
Example usage of the Thermal Mass Correction Neural Network
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from architecture import ThermalMassCorrectionNet, ThermalMassDataset, train_model, ThermalMassLoss

# For demonstration without pytorch
print("PyTorch Thermal Mass Correction Network - Usage Example")
print("=" * 60)

def prepare_mvp_data_for_training(mvpa):
    """
    Prepare MVP data for neural network training
    
    Args:
        mvpa: MVPAnalyzer object with loaded data
        
    Returns:
        mvp_data: Dictionary with paired down/up profiles
        ctd_data: Dictionary with CTD reference data
    """
    mvp_data = {
        'TEMP_down': [],
        'COND_down': [],
        'PRES_down': [],
        'TEMP_up': [],
        'COND_up': [],
        'PRES_up': []
    }
    
    ctd_data = {
        'SALT_down': [],
        'SALT_up': []
    }
    
    # Pair up profiles (assuming even indices are down, odd are up)
    n_profiles = len(mvpa.PRES_mvp)
    
    for i in range(0, n_profiles-1, 2):
        # Down profile (even index)
        down_idx = i
        up_idx = i + 1
        
        # Skip if not enough data
        if (len(mvpa.TEMP_mvp[down_idx]) < 100 or 
            len(mvpa.TEMP_mvp[up_idx]) < 100):
            continue
            
        # MVP data
        mvp_data['TEMP_down'].append(mvpa.TEMP_mvp[down_idx])
        mvp_data['COND_down'].append(mvpa.COND_mvp[down_idx])
        mvp_data['PRES_down'].append(mvpa.PRES_mvp[down_idx])
        
        mvp_data['TEMP_up'].append(mvpa.TEMP_mvp[up_idx])
        mvp_data['COND_up'].append(mvpa.COND_mvp[up_idx])
        mvp_data['PRES_up'].append(mvpa.PRES_mvp[up_idx])
        
        # CTD reference data
        if hasattr(mvpa, 'SALT_ctd') and len(mvpa.SALT_ctd) > up_idx:
            ctd_data['SALT_down'].append(mvpa.SALT_ctd[down_idx])
            ctd_data['SALT_up'].append(mvpa.SALT_ctd[up_idx])
        else:
            # If no CTD data, create dummy reference (not recommended for real training)
            import gsw
            salt_down = gsw.SP_from_C(mvpa.COND_mvp[down_idx], 
                                    mvpa.TEMP_mvp[down_idx], 
                                    mvpa.PRES_mvp[down_idx])
            salt_up = gsw.SP_from_C(mvpa.COND_mvp[up_idx], 
                                  mvpa.TEMP_mvp[up_idx], 
                                  mvpa.PRES_mvp[up_idx])
            ctd_data['SALT_down'].append(salt_down)
            ctd_data['SALT_up'].append(salt_up)
    
    return mvp_data, ctd_data


def create_training_pipeline():
    """
    Create complete training pipeline for thermal mass correction
    """
    
    print("\n1. Model Architecture")
    print("-" * 20)
    print("Input: 6 channels (T, C, P for down and up profiles)")
    print("Architecture: Bidirectional LSTM + Feed-forward layers")
    print("Output: 8 parameters (alpha0, alphaS, tau0, tauS for down/up)")
    print("Loss: MSE between corrected salinity and CTD reference")
    
    print("\n2. Garau et al. 2011 Correction Formula")
    print("-" * 40)
    print("C_corrected[i] = C[i] + α·τ·(dT/dt) - (C_corrected[i-1] - C[i-1])·exp(-dt/τ)")
    print("where:")
    print("  α = α₀ + αₛ·S  (amplitude parameter)")
    print("  τ = τ₀ + τₛ·S  (time constant)")
    
    print("\n3. Training Process")
    print("-" * 19)
    print("- Input profiles are normalized and padded to fixed length")
    print("- Network predicts 8 correction parameters")
    print("- Parameters are applied to correct conductivity")
    print("- Salinity is computed from corrected C, T, P")
    print("- Loss compares with CTD reference salinity")
    
    return True


def example_usage_with_mvpa(mvpa):
    """
    Example of how to use the network with MVPAnalyzer data
    
    Args:
        mvpa: MVPAnalyzer object with loaded data
    """
    try:
        import torch
        import torch.utils.data
        
        print("\n4. Data Preparation")
        print("-" * 19)
        
        # Prepare data
        mvp_data, ctd_data = prepare_mvp_data_for_training(mvpa)
        
        print(f"Number of profile pairs: {len(mvp_data['TEMP_down'])}")
        
        if len(mvp_data['TEMP_down']) == 0:
            print("No suitable profile pairs found!")
            return
            
        # Create dataset
        dataset = ThermalMassDataset(mvp_data, ctd_data, sequence_length=1000)
        
        # Split train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False
        )
        
        print("\n5. Model Training")
        print("-" * 16)
        
        # Initialize model
        model = ThermalMassCorrectionNet(
            sequence_length=1000,
            hidden_size=128,
            num_layers=2
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model (example with 1 epoch for demonstration)
        print("Starting training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=1, lr=0.001
        )
        
        print("Training completed!")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        
        return model, train_losses, val_losses
        
    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        return None, None, None


def demonstrate_correction_theory():
    """
    Demonstrate the theoretical basis of thermal mass correction
    """
    print("\n6. Thermal Mass Correction Theory")
    print("-" * 33)
    print("""
The thermal mass correction addresses the fact that the conductivity sensor
has a finite response time to temperature changes. When the sensor moves
through the water column, rapid temperature changes cause the conductivity
measurement to lag behind the true value.

Key equations from Garau et al. 2011:

1. Thermal mass correction:
   C_corrected = C_measured + thermal_term - exponential_decay_term
   
2. Thermal term:
   thermal_term = α(P,S) × τ(P,S) × (dT/dt)
   
3. Parameter dependencies:
   α(P,S) = α₀ + αₚ×P + αₛ×S  (amplitude)
   τ(P,S) = τ₀ + τₚ×P + τₛ×S  (time constant)

Where:
- α controls the magnitude of correction
- τ controls the response time
- P is pressure, S is salinity
- dT/dt is the temperature time derivative

The neural network learns these parameters by minimizing the difference
between corrected salinity and CTD reference measurements.
    """)


if __name__ == "__main__":
    # Demonstrate the architecture
    create_training_pipeline()
    demonstrate_correction_theory()
    
    print("\n7. Integration with MVPAnalyzer")
    print("-" * 31)
    print("""
To use with your MVP data:

from neural_th.architecture import ThermalMassCorrectionNet, prepare_mvp_data_for_training
import torch

# 1. Load your MVP data
mvpa = Analyzer('/path/to/mvp/data/')
mvpa.load_mvp_data()
mvpa.load_ctd_data('/path/to/ctd/data/')

# 2. Prepare data for training
mvp_data, ctd_data = prepare_mvp_data_for_training(mvpa)

# 3. Create and train model
model = ThermalMassCorrectionNet()
# ... training code ...

# 4. Apply corrections
with torch.no_grad():
    predicted_params = model(input_features)
    # Use predicted_params to correct your data
    """)
    
    print("\nExample complete! Check the architecture.py file for full implementation.")